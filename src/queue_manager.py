"""
MotionForge - queue_manager.py
Scene processing queue with real output structure:

  outputs/
    clips/
      scene_001.mp4
      scene_002.mp4
    logs/
      scene_001.json
      scene_002.json
    reports/
      scene_001.md
      scene_002.md
    project_manifest.json
    generation_report.md
    failed_scenes.json
    project_queue_state.json   ← checkpoint for resume

Queue state is saved after every scene for Colab reconnect resilience.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logger import get_logger, build_scene_logger
from .fallback_manager import FallbackManager
from .report_generator import save_scene_json_log, save_scene_md_report
from .shared.types import GenerationResult, SceneResult

log = get_logger("motionforge.queue_manager")

# Queue item status constants
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


def _scene_id_str(scene: dict[str, Any]) -> str:
    return str(scene.get("id", "000")).zfill(3)


class QueueManager:
    """
    Converts scenes into queue items and drives sequential processing.

    Output layout:
      outputs/clips/    — MP4 files
      outputs/logs/     — per-scene JSON logs
      outputs/reports/  — per-scene Markdown reports
    """

    def __init__(
        self,
        output_root: Path,
        fallback_manager: FallbackManager,
        project_root: Path,
        continue_on_error: bool = True,
        checkpoint_after_each: bool = True,
    ) -> None:
        self._output_root = output_root
        self._clips_dir = output_root / "clips"
        self._logs_dir = output_root / "logs"
        self._reports_dir = output_root / "reports"
        self._fallback_manager = fallback_manager
        self._project_root = project_root
        self._continue_on_error = continue_on_error
        self._checkpoint = checkpoint_after_each
        self._queue: list[dict[str, Any]] = []

    # ---------------------------------------------------------------------- #
    # Queue construction
    # ---------------------------------------------------------------------- #

    def build_queue(self, scenes: list[dict[str, Any]]) -> None:
        """Convert validated scene list into pending queue items."""
        self._queue = []
        for scene in scenes:
            sid_str = _scene_id_str(scene)
            output_filename = f"scene_{sid_str}.mp4"
            scene_hash = self._compute_scene_hash(scene)

            self._queue.append({
                "scene_id": scene.get("id"),
                "scene_name": scene.get("name"),
                "scene_hash": scene_hash,
                "status": STATUS_PENDING,
                "attempts": 0,
                "selected_model": scene.get("_resolved_model", "image_pan_zoom"),
                "output_path": str(self._clips_dir / output_filename),
                "error": None,
                "model_used": None,
                "start_time": None,
                "end_time": None,
                "generation_time_seconds": 0.0,
                # Video metadata fields
                "duration_seconds": None,
                "fps": None,
                "width": None,
                "height": None,
                "seed": None,
                "frame_count": None,
                "file_size_bytes": None,
                # Internal — not serialised to checkpoint
                "_scene_data": scene,
            })

        log.info("Queue built: %d item(s).", len(self._queue))

    # ---------------------------------------------------------------------- #
    # Checkpoint / resume support
    # ---------------------------------------------------------------------- #

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Restore status from a previously saved queue state.
        Returns True if checkpoint was loaded successfully.
        Scenes already in SUCCESS state will be skipped on the next run.
        """
        if not checkpoint_path.exists():
            return False

        with checkpoint_path.open("r", encoding="utf-8") as f:
            saved = json.load(f)

        by_id = {item["scene_id"]: item for item in self._queue}
        for saved_item in saved:
            sid = saved_item.get("scene_id")
            if sid in by_id:
                item = by_id[sid]
                
                # Smart Caching: Only restore SUCCESS state if the scene hash matches
                saved_hash = saved_item.get("scene_hash")
                saved_status = saved_item.get("status", STATUS_PENDING)
                
                if saved_hash == item["scene_hash"] and saved_status == STATUS_SUCCESS:
                    # Hash matches and it was a success. Restore fully.
                    item["status"] = saved_status
                    item["attempts"] = saved_item.get("attempts", 0)
                    item["output_path"] = saved_item.get("output_path", item["output_path"])
                    item["model_used"] = saved_item.get("model_used")
                    item["error"] = saved_item.get("error")
                    item["generation_time_seconds"] = saved_item.get("generation_time_seconds", 0.0)
                    item["duration_seconds"] = saved_item.get("duration_seconds")
                    item["fps"] = saved_item.get("fps")
                    item["width"] = saved_item.get("width")
                    item["height"] = saved_item.get("height")
                    item["seed"] = saved_item.get("seed")
                    item["frame_count"] = saved_item.get("frame_count")
                    item["file_size_bytes"] = saved_item.get("file_size_bytes")
                elif saved_hash != item["scene_hash"] and saved_status == STATUS_SUCCESS:
                    # Hash changed! Force a regeneration by leaving it as PENDING.
                    log.info("Scene '%s' config or image modified. Forcing regeneration.", item["scene_name"])
                else:
                    # It wasn't a success before anyway, leave it pending.
                    pass

        resumed = sum(1 for i in self._queue if i["status"] == STATUS_SUCCESS)
        log.info("Checkpoint loaded — %d scene(s) already completed.", resumed)
        return True

    def _save_checkpoint(self, checkpoint_path: Path) -> None:
        """Persist current queue state (without _scene_data)."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = [
            {k: v for k, v in item.items() if k != "_scene_data"}
            for item in self._queue
        ]
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)

    # ---------------------------------------------------------------------- #
    # Processing loop
    # ---------------------------------------------------------------------- #

    def process_all(self, checkpoint_path: Path) -> list[dict[str, Any]]:
        """
        Process all pending queue items sequentially.
        Returns list of result dicts (one per scene).
        """
        # Ensure output directories exist
        self._clips_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []

        for item in self._queue:
            if item["status"] == STATUS_SUCCESS:
                log.info("Scene '%s' already done — skipping.", item["scene_name"])
                results.append(self._item_to_result(item))
                continue

            scene = item["_scene_data"]
            output_path = Path(item["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Per-scene logger
            sid_str = str(item["scene_id"]).zfill(3)
            scene_logger = build_scene_logger(
                item["scene_name"],
                self._logs_dir / f"scene_{sid_str}_pipeline.txt",
            )

            item["status"] = STATUS_RUNNING
            item["start_time"] = datetime.now(timezone.utc).isoformat()

            # Run generation through fallback chain
            gen_result: GenerationResult = self._fallback_manager.run_scene(
                scene, output_path, scene_logger
            )

            item["end_time"] = datetime.now(timezone.utc).isoformat()
            item["attempts"] = gen_result.generation_time_seconds and 0  # overridden below
            item["model_used"] = gen_result.model_used
            item["error"] = gen_result.error_message
            item["generation_time_seconds"] = gen_result.generation_time_seconds
            item["status"] = STATUS_SUCCESS if gen_result.success else STATUS_FAILED

            if gen_result.success and gen_result.output_path:
                item["output_path"] = gen_result.output_path
                item["duration_seconds"] = gen_result.duration_seconds
                item["fps"] = gen_result.fps
                item["width"] = gen_result.width
                item["height"] = gen_result.height
                item["seed"] = gen_result.seed
                item["frame_count"] = gen_result.frame_count
                # Get file size from disk
                try:
                    item["file_size_bytes"] = Path(gen_result.output_path).stat().st_size
                except OSError:
                    item["file_size_bytes"] = None

                # Copy input image next to clip for reference
                self._copy_input_image(scene)

            # Build SceneResult and write per-scene reports
            scene_result = self._build_scene_result(item)
            save_scene_json_log(scene_result, self._logs_dir)
            save_scene_md_report(scene_result, self._reports_dir)

            # Save error file for failed scenes
            if item["status"] == STATUS_FAILED:
                self._write_failure_files(item, sid_str)

            if self._checkpoint:
                self._save_checkpoint(checkpoint_path)

            results.append(self._item_to_result(item))

            if item["status"] == STATUS_FAILED and not self._continue_on_error:
                log.error(
                    "continue_on_error=False — aborting after scene '%s'.",
                    item["scene_name"],
                )
                break

        return results

    # ---------------------------------------------------------------------- #
    # Accessors
    # ---------------------------------------------------------------------- #

    def get_queue(self) -> list[dict[str, Any]]:
        return self._queue

    def get_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self._queue:
            status = item.get("status", STATUS_PENDING)
            counts[status] = counts.get(status, 0) + 1
        return counts

    def get_output_dirs(self) -> dict[str, Path]:
        return {
            "clips": self._clips_dir,
            "logs": self._logs_dir,
            "reports": self._reports_dir,
        }

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _copy_input_image(self, scene: dict[str, Any]) -> None:
        """Copy input image to clips dir for archiving."""
        src = self._project_root / scene.get("input_image", "")
        if src.exists():
            dst = self._clips_dir / src.name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))

    def _write_failure_files(self, item: dict[str, Any], sid_str: str) -> None:
        """Write error text file for failed scenes into logs dir."""
        error_path = self._logs_dir / f"scene_{sid_str}_error.txt"
        with error_path.open("w", encoding="utf-8") as f:
            f.write(f"Scene: {item['scene_name']}\n")
            f.write(f"Status: {item['status']}\n")
            f.write(f"Attempts: {item.get('attempts', 0)}\n")
            f.write(f"Error: {item.get('error', 'unknown')}\n")
            f.write(f"Start: {item.get('start_time')}\n")
            f.write(f"End: {item.get('end_time')}\n")

    def _compute_scene_hash(self, scene: dict[str, Any]) -> str:
        """Compute MD5 hash of scene dict and input image metadata for smart caching."""
        clean_scene = {k: v for k, v in scene.items() if not k.startswith("_")}
        scene_str = json.dumps(clean_scene, sort_keys=True)
        hasher = hashlib.md5(scene_str.encode("utf-8"))
        
        image_rel = scene.get("input_image", "")
        if image_rel:
            image_path = self._project_root / image_rel
            if image_path.exists():
                stat = image_path.stat()
                hasher.update(f"{stat.st_size}_{stat.st_mtime}".encode("utf-8"))
        
        return hasher.hexdigest()

    @staticmethod
    def _build_scene_result(item: dict[str, Any]) -> SceneResult:
        return SceneResult(
            scene_id=item["scene_id"],
            scene_name=item["scene_name"],
            status=item["status"],
            model_used=item.get("model_used"),
            output_path=item.get("output_path"),
            attempts=item.get("attempts", 0),
            error=item.get("error"),
            start_time=item.get("start_time"),
            end_time=item.get("end_time"),
            generation_time_seconds=item.get("generation_time_seconds", 0.0),
            duration_seconds=item.get("duration_seconds"),
            fps=item.get("fps"),
            width=item.get("width"),
            height=item.get("height"),
            seed=item.get("seed"),
            frame_count=item.get("frame_count"),
            file_size_bytes=item.get("file_size_bytes"),
        )

    @staticmethod
    def _item_to_result(item: dict[str, Any]) -> dict[str, Any]:
        """Convert queue item to a plain dict for report_generator."""
        return {
            "scene_id": item["scene_id"],
            "scene_name": item["scene_name"],
            "status": item["status"],
            "model_used": item.get("model_used"),
            "output_path": item.get("output_path"),
            "attempts": item.get("attempts", 0),
            "error": item.get("error"),
            "start_time": item.get("start_time"),
            "end_time": item.get("end_time"),
            "generation_time_seconds": item.get("generation_time_seconds", 0.0),
            "duration_seconds": item.get("duration_seconds"),
            "fps": item.get("fps"),
            "width": item.get("width"),
            "height": item.get("height"),
            "seed": item.get("seed"),
            "frame_count": item.get("frame_count"),
            "file_size_bytes": item.get("file_size_bytes"),
        }

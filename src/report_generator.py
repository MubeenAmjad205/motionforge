"""
MotionForge - report_generator.py
Generates:
  - Per-scene JSON log  → outputs/logs/scene_NNN.json
  - Per-scene Markdown  → outputs/reports/scene_NNN.md
  - Project manifest    → outputs/project_manifest.json
  - Generation report   → outputs/generation_report.md
  - Failed scenes list  → outputs/failed_scenes.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logger import get_logger
from .shared.types import SceneResult

log = get_logger("motionforge.report_generator")


# --------------------------------------------------------------------------- #
# Per-scene JSON log
# --------------------------------------------------------------------------- #

def save_scene_json_log(result: SceneResult, logs_dir: Path) -> Path:
    """Write one JSON log file per scene to outputs/logs/."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    scene_id_str = str(result.scene_id).zfill(3)
    path = logs_dir / f"scene_{scene_id_str}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    log.debug("Scene log saved → %s", path)
    return path


# --------------------------------------------------------------------------- #
# Per-scene Markdown report
# --------------------------------------------------------------------------- #

def save_scene_md_report(result: SceneResult, reports_dir: Path) -> Path:
    """Write one Markdown report per scene to outputs/reports/."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    scene_id_str = str(result.scene_id).zfill(3)
    path = reports_dir / f"scene_{scene_id_str}.md"

    status_emoji = "✅" if result.status == "success" else "❌"
    lines: list[str] = [
        f"# Scene {result.scene_id} — {result.scene_name}\n",
        f"**Status**: {status_emoji} `{result.status}`  ",
        f"**Model**: `{result.model_used or '—'}`  ",
        f"**Attempts**: {result.attempts}  ",
        f"**Start**: {result.start_time or '—'}  ",
        f"**End**: {result.end_time or '—'}  ",
        f"**Generation time**: {result.generation_time_seconds:.1f}s  \n",
    ]

    if result.status == "success" and result.output_path:
        lines += [
            "## Output\n",
            f"**File**: `{result.output_path}`  ",
            f"**Duration**: {result.duration_seconds}s  ",
            f"**FPS**: {result.fps}  ",
            f"**Resolution**: {result.width}×{result.height}  ",
            f"**Frames**: {result.frame_count}  ",
            f"**Seed**: {result.seed}  ",
            f"**File size**: {_fmt_bytes(result.file_size_bytes)}  \n",
        ]
    else:
        lines += [
            "## Failure Details\n",
            f"**Error**: `{result.error or 'unknown'}`  \n",
        ]

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.debug("Scene report saved → %s", path)
    return path


# --------------------------------------------------------------------------- #
# Project manifest (JSON)
# --------------------------------------------------------------------------- #

def build_manifest(
    project: dict[str, Any],
    results: list[dict[str, Any]],
    zip_path: Path | None,
) -> dict[str, Any]:
    """Build the project_manifest.json structure from raw result dicts."""
    total = len(results)
    successes = sum(1 for r in results if r.get("status") == "success")
    failures = sum(1 for r in results if r.get("status") == "failed")

    scenes_summary: list[dict[str, Any]] = []
    for r in results:
        entry: dict[str, Any] = {
            "id": r["scene_id"],
            "name": r["scene_name"],
            "status": r["status"],
            "model_used": r.get("model_used"),
            "output": r.get("output_path"),
            "attempts": r.get("attempts", 0),
            "error": r.get("error"),
            "start_time": r.get("start_time"),
            "end_time": r.get("end_time"),
            "generation_time_seconds": r.get("generation_time_seconds", 0),
        }
        # Include video metadata if success
        if r.get("status") == "success" and r.get("output_path"):
            entry["video"] = {
                "duration_seconds": r.get("duration_seconds"),
                "fps": r.get("fps"),
                "width": r.get("width"),
                "height": r.get("height"),
                "seed": r.get("seed"),
                "frame_count": r.get("frame_count"),
                "file_size_bytes": r.get("file_size_bytes"),
            }
        scenes_summary.append(entry)

    return {
        "project_name": project.get("project", {}).get("name", "unknown"),
        "project_version": project.get("project", {}).get("version", "1.0.0"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_scenes": total,
        "successful_scenes": successes,
        "failed_scenes": failures,
        "success_rate_pct": round(successes / total * 100, 1) if total else 0.0,
        "output_zip": str(zip_path) if zip_path else None,
        "scenes": scenes_summary,
    }


def save_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "project_manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest saved → %s", path)
    return path


# --------------------------------------------------------------------------- #
# Failed scenes JSON
# --------------------------------------------------------------------------- #

def save_failed_scenes(results: list[dict[str, Any]], output_dir: Path) -> Path | None:
    failed = [r for r in results if r.get("status") == "failed"]
    if not failed:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "failed_scenes.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2)
    log.info("Failed scenes saved → %s (%d failed)", path, len(failed))
    return path


# --------------------------------------------------------------------------- #
# Global Markdown report
# --------------------------------------------------------------------------- #

def build_markdown_report(manifest: dict[str, Any]) -> str:
    """Build the global generation_report.md content string."""
    lines: list[str] = [
        "# MotionForge — Generation Report\n",
        f"**Project**: {manifest['project_name']}  ",
        f"**Version**: {manifest['project_version']}  ",
        f"**Generated**: {manifest['generated_at']}  \n",
        "## Summary\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Scenes | {manifest['total_scenes']} |",
        f"| Successful | {manifest['successful_scenes']} |",
        f"| Failed | {manifest['failed_scenes']} |",
        f"| Success Rate | {manifest['success_rate_pct']}% |",
        f"| Output ZIP | `{manifest.get('output_zip') or '—'}` |",
        "",
        "## Scene Results\n",
        "| # | Name | Status | Model | Duration | Resolution | Gen Time | Attempts |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for scene in manifest.get("scenes", []):
        video = scene.get("video", {}) or {}
        duration = f"{video.get('duration_seconds', '—')}s" if video.get("duration_seconds") else "—"
        res = f"{video.get('width', '—')}x{video.get('height', '—')}" if video else "—"
        gen_time = f"{scene.get('generation_time_seconds', 0):.1f}s"
        status_emoji = "✅" if scene["status"] == "success" else "❌"
        lines.append(
            f"| {scene['id']} | {scene['name']} "
            f"| {status_emoji} {scene['status']} "
            f"| `{scene.get('model_used') or '—'}` "
            f"| {duration} | {res} | {gen_time} | {scene.get('attempts', 0)} |"
        )

    lines += ["", "## Failed Scenes\n"]
    failed = [s for s in manifest.get("scenes", []) if s["status"] == "failed"]
    if not failed:
        lines.append("_No scenes failed._")
    else:
        for s in failed:
            lines += [
                f"### Scene {s['id']}: {s['name']}",
                f"**Error**: `{s.get('error') or 'unknown'}`  ",
                f"**Attempts**: {s.get('attempts', 0)}  \n",
            ]

    lines += ["\n---\n", "_Generated by MotionForge pipeline._"]
    return "\n".join(lines)


def save_markdown_report(report: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "generation_report.md"
    with path.open("w", encoding="utf-8") as f:
        f.write(report)
    log.info("Markdown report saved → %s", path)
    return path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fmt_bytes(size: int | None) -> str:
    if size is None:
        return "—"
    if size < 1024:
        return f"{size} B"
    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 ** 2):.2f} MB"

"""
MotionForge - video_generator.py
High-level entry point for single-scene video generation.
Bridges QueueManager and the model adapter pipeline.
Used by main.py and the Colab notebook.
"""

from pathlib import Path
from typing import Any

from .logger import get_logger, build_scene_logger
from .fallback_manager import FallbackManager
from .postprocess import get_video_info

log = get_logger("motionforge.video_generator")


def generate_scene_video(
    scene: dict[str, Any],
    output_path: Path,
    fallback_manager: FallbackManager,
) -> dict[str, Any]:
    """
    Generate a single scene video using the FallbackManager.

    Parameters
    ----------
    scene           : Fully resolved scene dict.
    output_path     : Destination path for the output MP4.
    fallback_manager: Configured FallbackManager instance.

    Returns
    -------
    Result dict containing status, output_path, model_used, attempts, error.
    """
    scene_dir = output_path.parent
    scene_dir.mkdir(parents=True, exist_ok=True)

    scene_id_str = str(scene.get("id", "unknown")).zfill(3)
    scene_logger = build_scene_logger(
        scene.get("name", "scene"),
        scene_dir / f"scene_{scene_id_str}_log.txt",
    )

    result = fallback_manager.run_scene(scene, output_path, scene_logger)

    if result["status"] == "success" and result.get("output_path"):
        info = get_video_info(Path(result["output_path"]))
        result["video_info"] = info
        log.info(
            "Scene '%s' → %s | %dx%d | %.2fs | %dfps",
            scene.get("name"),
            result["output_path"],
            info.get("width", 0),
            info.get("height", 0),
            info.get("duration", 0),
            info.get("fps", 0),
        )

    return result

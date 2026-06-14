"""
MotionForge - scene_loader.py
Loads project JSON and merges global defaults into each scene definition.
"""

import json
from pathlib import Path
from typing import Any

from .logger import get_logger

log = get_logger("motionforge.scene_loader")

# Fields that can be overridden per-scene from global_settings
_OVERRIDABLE_FIELDS = (
    "duration_seconds",
    "fps",
    "resolution",
    "seed",
    "motion_strength",
    "guidance_scale",
    "num_inference_steps",
    "output_format",
    "video_codec",
)


def load_project(scenes_json_path: Path) -> dict[str, Any]:
    """
    Read and parse the project scenes JSON.
    Returns the raw project dict.
    Raises ValueError on parse failure.
    """
    if not scenes_json_path.exists():
        raise FileNotFoundError(f"Scenes file not found: {scenes_json_path}")

    with scenes_json_path.open("r", encoding="utf-8") as f:
        try:
            project = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {scenes_json_path}: {exc}") from exc

    log.info("Loaded project '%s' with %d scene(s).", project.get("project", {}).get("name", "—"), len(project.get("scenes", [])))
    return project


def merge_defaults(scene: dict[str, Any], global_settings: dict[str, Any]) -> dict[str, Any]:
    """
    Return a new scene dict with global_settings values used as defaults
    for any fields the scene has not explicitly set.
    """
    merged = dict(scene)
    for field in _OVERRIDABLE_FIELDS:
        if field not in merged or merged[field] is None:
            if field in global_settings:
                merged[field] = global_settings[field]

    # Model: respect model_override, then global default_model
    if not merged.get("model_override"):
        merged["_resolved_model"] = global_settings.get("default_model", "mock_adapter")
    else:
        merged["_resolved_model"] = merged["model_override"]

    return merged


def load_scenes(project: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract, filter (enabled only), and merge scenes with global defaults.
    Returns list of fully-resolved scene dicts.
    """
    global_settings = project.get("global_settings", {})
    raw_scenes = project.get("scenes", [])

    resolved: list[dict[str, Any]] = []
    for raw_scene in raw_scenes:
        if not raw_scene.get("enabled", True):
            log.info("Skipping disabled scene: %s", raw_scene.get("name", raw_scene.get("id")))
            continue
        resolved.append(merge_defaults(raw_scene, global_settings))

    log.info("Resolved %d enabled scene(s).", len(resolved))
    return resolved

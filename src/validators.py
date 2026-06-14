"""
MotionForge - validators.py
Schema validation for scenes and output video files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .logger import get_logger

log = get_logger("motionforge.validators")

_MIN_FILE_SIZE_BYTES = 10_000

_REQUIRED_SCENE_FIELDS = ("id", "name", "input_image", "motion_prompt")

_SUPPORTED_MOTION_EFFECTS = (
    "zoom_in", "zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
    "zoom_in_pan_right", "zoom_in_pan_left",
    "zoom_out_pan_right", "zoom_out_pan_left",
    "static",
)


# --------------------------------------------------------------------------- #
# Scene validation
# --------------------------------------------------------------------------- #

def validate_scene(
    scene: dict[str, Any],
    project_root: Path,
    registered_models: Optional[list[str]] = None,
) -> tuple[bool, list[str]]:
    """
    Validate a single scene dict.
    Returns (is_valid, list_of_error_strings).

    Parameters
    ----------
    scene            : Resolved scene dict.
    project_root     : Root path used to resolve relative image paths.
    registered_models: If provided, validate model_override against this list.
    """
    errors: list[str] = []

    # Required fields
    for field in _REQUIRED_SCENE_FIELDS:
        val = scene.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"Missing or empty required field: '{field}'")

    # Scene id must be an integer
    scene_id = scene.get("id")
    if scene_id is not None and not isinstance(scene_id, int):
        errors.append(f"'id' must be an integer, got: {type(scene_id).__name__}")

    # Input image must exist
    image_rel = scene.get("input_image", "")
    if image_rel:
        image_path = project_root / image_rel
        if not image_path.exists():
            errors.append(f"Input image not found: {image_path}")
        elif image_path.stat().st_size == 0:
            errors.append(f"Input image is empty (0 bytes): {image_path}")

    # FPS range
    fps = scene.get("fps")
    if fps is not None:
        if not isinstance(fps, (int, float)):
            errors.append(f"'fps' must be a number, got: {type(fps).__name__}")
        elif not (1 <= fps <= 120):
            errors.append(f"'fps' out of range [1, 120]: {fps}")

    # Duration range
    duration = scene.get("duration_seconds")
    if duration is not None:
        if not isinstance(duration, (int, float)):
            errors.append(f"'duration_seconds' must be a number, got: {type(duration).__name__}")
        elif not (0.5 <= duration <= 120):
            errors.append(f"'duration_seconds' out of range [0.5, 120]: {duration}")

    # Resolution
    resolution = scene.get("resolution", {})
    if resolution:
        width = resolution.get("width")
        height = resolution.get("height")
        if width is not None and not (64 <= width <= 3840):
            errors.append(f"Resolution width out of range [64, 3840]: {width}")
        if height is not None and not (64 <= height <= 2160):
            errors.append(f"Resolution height out of range [64, 2160]: {height}")

    # Seed must be int or null
    seed = scene.get("seed")
    if seed is not None and not isinstance(seed, int):
        errors.append(
            f"'seed' must be an integer or null, got: {type(seed).__name__} ({seed!r})"
        )

    # Model override — validate against registry if provided
    model_override = scene.get("model_override")
    if model_override is not None:
        if not isinstance(model_override, str) or not model_override.strip():
            errors.append(f"'model_override' must be a non-empty string or null, got: {model_override!r}")
        elif registered_models and model_override not in registered_models:
            errors.append(
                f"'model_override' references unknown model '{model_override}'. "
                f"Registered: {registered_models}"
            )

    # motion_effect — warn but don't fail (unknown effects fall back gracefully)
    effect = scene.get("motion_effect")
    if effect is not None and effect not in _SUPPORTED_MOTION_EFFECTS:
        log.warning(
            "Scene '%s': unknown motion_effect '%s'. Will use default. Supported: %s",
            scene.get("name"), effect, _SUPPORTED_MOTION_EFFECTS,
        )

    return len(errors) == 0, errors


def validate_all_scenes(
    scenes: list[dict[str, Any]],
    project_root: Path,
    registered_models: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Validate all scenes. Skips invalid scenes with detailed error logs.
    Returns only valid scenes.
    """
    valid: list[dict[str, Any]] = []
    for scene in scenes:
        ok, errors = validate_scene(scene, project_root, registered_models)
        if ok:
            valid.append(scene)
        else:
            log.error(
                "Scene '%s' (id=%s) failed validation — skipping.\n  Errors:\n  - %s",
                scene.get("name"),
                scene.get("id"),
                "\n  - ".join(errors),
            )
    log.info("%d/%d scene(s) passed validation.", len(valid), len(scenes))
    return valid


# --------------------------------------------------------------------------- #
# Output video validation
# --------------------------------------------------------------------------- #

def validate_video_output(
    path: Path,
    expected_duration: Optional[float] = None,
    min_size_bytes: int = _MIN_FILE_SIZE_BYTES,
) -> tuple[bool, str]:
    """
    Verify a generated video file is usable.
    Returns (is_valid, message).
    """
    if not path.exists():
        return False, f"File does not exist: {path}"

    size = path.stat().st_size
    if size < min_size_bytes:
        return False, f"File too small ({size} bytes < {min_size_bytes}): {path}"

    try:
        from moviepy.editor import VideoFileClip  # type: ignore

        clip = VideoFileClip(str(path))
        actual_duration = clip.duration
        clip.close()

        if actual_duration is None or actual_duration <= 0:
            return False, f"Video has zero or unknown duration: {path}"

        if expected_duration is not None:
            tolerance = max(1.0, expected_duration * 0.25)
            if abs(actual_duration - expected_duration) > tolerance:
                return (
                    False,
                    f"Duration mismatch — expected ~{expected_duration:.1f}s, "
                    f"got {actual_duration:.1f}s",
                )
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not open video with MoviePy: {exc}"

    return True, "OK"

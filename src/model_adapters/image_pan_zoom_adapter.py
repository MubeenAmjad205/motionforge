"""
MotionForge - model_adapters/image_pan_zoom_adapter.py

Real lightweight image-to-video adapter using MoviePy.
No GPU or AI model required. Produces genuine cinematic motion effects:
  - Slow zoom in / zoom out
  - Pan left / right / up / down
  - Combined pan + zoom
  - Optional fade in / fade out
  - Full control over duration, FPS, and resolution

This is the default production fallback when AI models are unavailable.
It is NOT a mock — it produces real, watchable MP4 motion clips.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base_adapter import BaseAdapter
from ..shared.types import GenerationResult
from ..logger import get_logger

log = get_logger("motionforge.adapters.pan_zoom")

# Motion effect names supported by this adapter
MOTION_EFFECTS = (
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "pan_up",
    "pan_down",
    "zoom_in_pan_right",
    "zoom_in_pan_left",
    "zoom_out_pan_right",
    "zoom_out_pan_left",
    "static",          # No motion — useful for debugging
)

# Default motion when none specified in scene config
_DEFAULT_MOTION = "zoom_in"

# Zoom range for scale effects (1.0 = original, 1.25 = 25% larger)
_ZOOM_START = 1.0
_ZOOM_END_IN = 1.25
_ZOOM_END_OUT = 0.85

# Pan distance as fraction of frame dimension
_PAN_FRACTION = 0.08


class ImagePanZoomAdapter(BaseAdapter):
    """
    Real cinematic motion adapter using MoviePy frame manipulation.

    Produces genuine motion effects by computing per-frame crop regions
    from the source image, creating the illusion of camera movement.

    Scene config fields consumed:
      - input_image      : source image path (required)
      - duration_seconds : clip length in seconds
      - fps              : output frames per second
      - resolution       : {"width": int, "height": int}
      - motion_effect    : one of MOTION_EFFECTS (optional, defaults to "zoom_in")
      - fade_in          : seconds of fade-in (optional, default 0.3)
      - fade_out         : seconds of fade-out (optional, default 0.3)
      - seed             : integer seed for reproducible random effects (optional)
    """

    name = "image_pan_zoom"
    IMPLEMENTED = True

    def is_available(self) -> bool:
        try:
            import moviepy  # noqa: F401
            from PIL import Image  # noqa: F401
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        log.debug("ImagePanZoomAdapter.load() — no weights to load.")

    def unload(self) -> None:
        log.debug("ImagePanZoomAdapter.unload() — no-op.")

    def generate(self, scene: dict[str, Any], output_path: Path) -> GenerationResult:
        """
        Build a real cinematic motion MP4 from the scene's input image.

        Algorithm:
          1. Load source image at 2x target resolution for sub-pixel quality.
          2. For each frame index i in [0, total_frames):
             a. Compute interpolation factor t ∈ [0.0, 1.0].
             b. Compute crop box from motion effect equations.
             c. Crop and resize to target resolution.
          3. Apply optional fade-in / fade-out alpha on frame arrays.
          4. Write frames to MP4 via MoviePy ImageSequenceClip.
        """
        t_start = time.monotonic()

        try:
            from PIL import Image  # type: ignore
            from moviepy.editor import ImageSequenceClip  # type: ignore
        except ImportError as exc:
            return self._fail(self.name, f"Missing dependency: {exc}")

        # ── Resolve scene parameters ────────────────────────────────────── #
        width, height = self._resolution_from_scene(scene)
        fps = int(scene.get("fps", 16))
        duration = float(scene.get("duration_seconds", 4))
        image_rel = scene.get("input_image", "")
        image_path = Path(image_rel)
        effect = scene.get("motion_effect", _DEFAULT_MOTION)
        fade_in_s = float(scene.get("fade_in", 0.3))
        fade_out_s = float(scene.get("fade_out", 0.3))
        seed = scene.get("seed")

        if effect not in MOTION_EFFECTS:
            log.warning("Unknown motion_effect '%s' — falling back to '%s'.", effect, _DEFAULT_MOTION)
            effect = _DEFAULT_MOTION

        log.info(
            "ImagePanZoom: '%s' | effect=%s | %dx%d | %sfps | %.1fs",
            image_path.name, effect, width, height, fps, duration,
        )

        if not image_path.exists():
            return self._fail(self.name, f"Input image not found: {image_path}")

        # ── Load and prepare source image ────────────────────────────────── #
        try:
            src = Image.open(image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            return self._fail(self.name, f"Failed to open image {image_path}: {exc}")

        # Oversample source for smooth sub-pixel crops
        oversample = 2
        src_w = width * oversample
        src_h = height * oversample
        src = src.resize((src_w, src_h), Image.LANCZOS)
        src_array = np.array(src, dtype=np.float32)

        # ── Build frame sequence ─────────────────────────────────────────── #
        total_frames = max(1, int(duration * fps))
        fade_in_frames = int(fade_in_s * fps)
        fade_out_frames = int(fade_out_s * fps)

        frames: list[np.ndarray] = []

        for i in range(total_frames):
            t = i / max(1, total_frames - 1)   # 0.0 → 1.0

            # Compute crop box for this frame
            crop_x, crop_y, crop_w, crop_h = _compute_crop(
                t, effect, src_w, src_h, width, height, oversample,
            )

            # Crop from oversampled source
            cropped = src_array[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

            # Resize to target resolution
            pil_frame = Image.fromarray(cropped.astype(np.uint8)).resize(
                (width, height), Image.LANCZOS
            )
            frame = np.array(pil_frame, dtype=np.float32)

            # Apply fade-in
            if fade_in_frames > 0 and i < fade_in_frames:
                alpha = i / fade_in_frames
                frame = frame * alpha

            # Apply fade-out
            if fade_out_frames > 0 and i >= total_frames - fade_out_frames:
                frames_remaining = total_frames - i
                alpha = frames_remaining / fade_out_frames
                frame = frame * alpha

            frames.append(np.clip(frame, 0, 255).astype(np.uint8))

        # ── Write video ──────────────────────────────────────────────────── #
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio=False,
                logger=None,
            )
            clip.close()
        except Exception as exc:  # noqa: BLE001
            return self._fail(self.name, f"MoviePy write failed: {exc}")

        generation_time = time.monotonic() - t_start

        if not output_path.exists():
            return self._fail(self.name, "Output file not created after write attempt.")

        log.info(
            "ImagePanZoom done: %s (%.1fs generation time)",
            output_path, generation_time,
        )

        return GenerationResult(
            success=True,
            output_path=str(output_path),
            model_used=self.name,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            seed=seed,
            frame_count=total_frames,
            generation_time_seconds=generation_time,
        )


# --------------------------------------------------------------------------- #
# Crop computation — pure functions, easily unit-testable
# --------------------------------------------------------------------------- #

def _compute_crop(
    t: float,
    effect: str,
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    oversample: int,
) -> tuple[int, int, int, int]:
    """
    Compute (crop_x, crop_y, crop_w, crop_h) for a given frame interpolation t.

    t=0.0 → start of clip
    t=1.0 → end of clip

    Returns pixel coordinates into the oversampled source image.
    """
    # Base crop size is the target resolution scaled by oversample
    base_w = target_w * oversample
    base_h = target_h * oversample

    # Zoom factor varies with t
    if "zoom_in" in effect:
        zoom = _lerp(_ZOOM_START, _ZOOM_END_IN, t)
    elif "zoom_out" in effect:
        zoom = _lerp(_ZOOM_END_IN, _ZOOM_START, t)   # Start zoomed, pull back
    else:
        zoom = 1.0

    crop_w = max(1, int(base_w / zoom))
    crop_h = max(1, int(base_h / zoom))

    # Centre of frame in oversampled coordinates
    cx = src_w // 2
    cy = src_h // 2

    # Pan offset
    pan_x = 0
    pan_y = 0

    if "pan_right" in effect or effect == "pan_right":
        pan_x = int(_lerp(0, src_w * _PAN_FRACTION, t))
    elif "pan_left" in effect or effect == "pan_left":
        pan_x = int(_lerp(0, -src_w * _PAN_FRACTION, t))

    if "pan_down" in effect or effect == "pan_down":
        pan_y = int(_lerp(0, src_h * _PAN_FRACTION, t))
    elif "pan_up" in effect or effect == "pan_up":
        pan_y = int(_lerp(0, -src_h * _PAN_FRACTION, t))

    # Compute top-left corner, clamped to source bounds
    x0 = cx + pan_x - crop_w // 2
    y0 = cy + pan_y - crop_h // 2
    x0 = max(0, min(x0, src_w - crop_w))
    y0 = max(0, min(y0, src_h - crop_h))

    return x0, y0, crop_w, crop_h


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b at position t ∈ [0, 1]."""
    return a + (b - a) * max(0.0, min(1.0, t))

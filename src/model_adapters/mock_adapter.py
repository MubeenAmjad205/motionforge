"""
MotionForge - model_adapters/mock_adapter.py

Development/testing adapter ONLY.
Produces a brightness-pulsing MP4 from the input image — no real motion.
Used to verify pipeline plumbing without any real generation.

DO NOT set this as default_model in production configs.
Use image_pan_zoom_adapter for the real CPU fallback.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base_adapter import BaseAdapter
from ..shared.types import GenerationResult
from ..logger import get_logger

log = get_logger("motionforge.adapters.mock")


class MockAdapter(BaseAdapter):
    """
    Development-only adapter. Simulates generation with a brightness pulse.

    Use case: verifying pipeline structure without caring about video quality.
    Enable by setting model_override: "mock_adapter" on specific test scenes.
    Never default to this in production configs.
    """

    name = "mock_adapter"
    IMPLEMENTED = True

    def is_available(self) -> bool:
        try:
            import moviepy  # noqa: F401
            from PIL import Image  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        log.debug("MockAdapter.load() — no-op (development adapter).")

    def unload(self) -> None:
        log.debug("MockAdapter.unload() — no-op.")

    def generate(self, scene: dict[str, Any], output_path: Path) -> GenerationResult:
        """
        Build an MP4 by repeating the input image with a brightness pulse.
        This is NOT motion generation — it is pipeline verification only.
        """
        t_start = time.monotonic()

        try:
            from PIL import Image  # type: ignore
            from moviepy.editor import ImageSequenceClip  # type: ignore
        except ImportError as exc:
            return self._fail(self.name, f"Missing dependency: {exc}")

        width, height = self._resolution_from_scene(scene)
        fps = int(scene.get("fps", 16))
        duration = float(scene.get("duration_seconds", 4))
        image_path = Path(scene.get("input_image", ""))
        seed = scene.get("seed")

        log.info(
            "MockAdapter (DEV ONLY): %s | %dx%d | %dfps | %.1fs",
            image_path.name, width, height, fps, duration,
        )

        if not image_path.exists():
            return self._fail(self.name, f"Input image not found: {image_path}")

        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
        except Exception as exc:  # noqa: BLE001
            return self._fail(self.name, f"Failed to open image: {exc}")

        base_array = np.array(img, dtype=np.float32)
        total_frames = max(1, int(duration * fps))
        frames: list[np.ndarray] = []

        for i in range(total_frames):
            phase = i / max(1, total_frames - 1)
            factor = 1.0 + 0.12 * float(np.sin(phase * np.pi * 2))
            frame = np.clip(base_array * factor, 0, 255).astype(np.uint8)
            frames.append(frame)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(str(output_path), codec="libx264", audio=False, logger=None)
            clip.close()
        except Exception as exc:  # noqa: BLE001
            return self._fail(self.name, f"MoviePy write failed: {exc}")

        generation_time = time.monotonic() - t_start

        if not output_path.exists():
            return self._fail(self.name, "Output file not created after write attempt.")

        log.info("MockAdapter wrote: %s (%.1fs)", output_path, generation_time)

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

"""
MotionForge - model_adapters/framepack_adapter.py
Placeholder adapter for FramePack low-VRAM workflow.
NOT IMPLEMENTED — returns False from is_available() so the pipeline skips it cleanly.

Integration guide (Phase 2):
  Library:  https://github.com/lllyasviel/FramePack
  VRAM:     ~6 GB
  Frames:   multiples of 8 preferred
  Max res:  854x480 for free Colab

  Quick start:
    !git clone https://github.com/lllyasviel/FramePack
    !pip install -r FramePack/requirements.txt
    from framepack import FramePackPipeline
    pipe = FramePackPipeline(model_dir="models/framepack_low")
    video_frames = pipe.run(image=image, prompt=prompt, n_frames=n)

  Once implemented:
    1. Set IMPLEMENTED = True
    2. Implement is_available() checking framepack + torch
    3. Implement load(), generate(), unload()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_adapter import BaseAdapter
from ..shared.types import GenerationResult
from ..logger import get_logger

log = get_logger("motionforge.adapters.framepack")


class FramePackAdapter(BaseAdapter):
    """FramePack low-VRAM adapter — NOT YET IMPLEMENTED."""

    name = "framepack_low_vram"
    IMPLEMENTED = False   # Set True only when generate() is fully working

    def is_available(self) -> bool:
        # Always return False until fully implemented
        return False

    def load(self) -> None:
        raise NotImplementedError(
            "FramePackAdapter is not implemented. "
            "See module docstring for integration instructions."
        )

    def generate(self, scene: dict[str, Any], output_path: Path) -> GenerationResult:
        raise NotImplementedError("FramePackAdapter.generate() not yet implemented.")

    def unload(self) -> None:
        self._clear_gpu_cache()
        log.debug("FramePackAdapter.unload() called (no-op — not loaded).")

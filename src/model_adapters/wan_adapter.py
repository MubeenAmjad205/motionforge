"""
MotionForge - model_adapters/wan_adapter.py
Placeholder adapter for Wan2.1 I2V GGUF Q4.
NOT IMPLEMENTED — returns False from is_available() so the pipeline skips it cleanly.

Integration guide (Phase 2):
  Model:    Wan-AI/Wan2.1-I2V-14B-480P-GGUF (or 720P)
  Library:  wan2video (or compatible loader)
  VRAM:     ~6 GB Q2, ~12 GB Q4
  Frames:   16–81
  Max res:  1280x720

  Quick start:
    from wan2video import Wan2Pipeline
    pipe = Wan2Pipeline.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-GGUF")
    pipe.enable_model_cpu_offload()
    video = pipe(
        image=image,
        prompt=scene["motion_prompt"],
        negative_prompt=scene.get("negative_prompt", ""),
        num_frames=frame_count,
        guidance_scale=scene.get("guidance_scale", 6.0),
        seed=scene.get("seed"),
    ).frames

  Once implemented:
    1. Set IMPLEMENTED = True
    2. Implement is_available() with real torch + library checks
    3. Implement load(), generate(), unload()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_adapter import BaseAdapter
from ..shared.types import GenerationResult
from ..logger import get_logger

log = get_logger("motionforge.adapters.wan")


class WanAdapter(BaseAdapter):
    """Wan2.1 I2V GGUF adapter — NOT YET IMPLEMENTED."""

    name = "wan2_1_i2v_gguf_q4"
    IMPLEMENTED = False   # Set True only when generate() is fully working

    def is_available(self) -> bool:
        # Always return False until fully implemented
        return False

    def load(self) -> None:
        raise NotImplementedError(
            "WanAdapter is not implemented. "
            "See module docstring for integration instructions."
        )

    def generate(self, scene: dict[str, Any], output_path: Path) -> GenerationResult:
        raise NotImplementedError("WanAdapter.generate() not yet implemented.")

    def unload(self) -> None:
        self._clear_gpu_cache()
        log.debug("WanAdapter.unload() called (no-op — not loaded).")

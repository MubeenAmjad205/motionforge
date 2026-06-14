"""
MotionForge - model_adapters/svd_adapter.py
Placeholder adapter for Stable Video Diffusion XT.
NOT IMPLEMENTED — returns False from is_available() so the pipeline skips it cleanly.

Integration guide (Phase 2):
  Model:    stabilityai/stable-video-diffusion-img2vid-xt
  Library:  diffusers >= 0.27
  VRAM:     ~8 GB fp16
  Frames:   14 or 25 only
  Max res:  1024x576

  Quick start:
    from diffusers import StableVideoDiffusionPipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16, variant="fp16",
    )
    pipe.enable_model_cpu_offload()
    frames = pipe(image, num_inference_steps=25, motion_bucket_id=127).frames[0]

  Once implemented:
    1. Set IMPLEMENTED = True
    2. Replace is_available() with real checks
    3. Implement load(), generate(), unload()
    4. Update configs/models.json notes field
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_adapter import BaseAdapter
from ..shared.types import GenerationResult
from ..logger import get_logger

log = get_logger("motionforge.adapters.svd")


class SVDAdapter(BaseAdapter):
    """Stable Video Diffusion XT adapter — NOT YET IMPLEMENTED."""

    name = "stable_video_diffusion_xt"
    IMPLEMENTED = False   # Set True only when generate() is fully working

    def is_available(self) -> bool:
        # Always return False until fully implemented
        return False

    def load(self) -> None:
        raise NotImplementedError(
            "SVDAdapter is not implemented. "
            "See module docstring for integration instructions."
        )

    def generate(self, scene: dict[str, Any], output_path: Path) -> GenerationResult:
        raise NotImplementedError("SVDAdapter.generate() not yet implemented.")

    def unload(self) -> None:
        self._clear_gpu_cache()
        log.debug("SVDAdapter.unload() called (no-op — not loaded).")

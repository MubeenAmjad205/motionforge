"""
MotionForge - model_adapters/svd_adapter.py
Adapter for Stable Video Diffusion XT using Hugging Face Diffusers.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base_adapter import BaseAdapter
from ..shared.types import GenerationResult
from ..logger import get_logger

log = get_logger("motionforge.adapters.svd")


class SVDAdapter(BaseAdapter):
    """Stable Video Diffusion XT adapter using diffusers."""

    name = "stable_video_diffusion_xt"
    IMPLEMENTED = True

    def __init__(self) -> None:
        self.pipe = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load(self) -> None:
        if self.pipe is not None:
            return

        import torch
        from diffusers import StableVideoDiffusionPipeline

        log.info("Loading SVD XT weights (stabilityai/stable-video-diffusion-img2vid-xt)...")
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        # Aggressive VRAM optimizations for 16GB GPUs (Colab T4)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()
            
        log.info("SVD XT loaded with max VRAM optimizations (offload, attention slicing).")

    def generate(self, scene: dict[str, Any], output_path: Path) -> GenerationResult:
        if self.pipe is None:
            return self._fail(self.name, "Pipeline not loaded. Call load() first.")

        import torch
        from PIL import Image
        import numpy as np
        from diffusers.utils import export_to_video

        t_start = time.monotonic()

        image_path = Path(scene.get("input_image", ""))
        if not image_path.exists():
            return self._fail(self.name, f"Image not found: {image_path}")

        # Ensure dimensions are multiples of 64
        width, height = self._resolution_from_scene(scene)
        width = (width // 64) * 64
        height = (height // 64) * 64
        
        # SVD-XT is optimized for 1024x576 or 576x1024
        # We will bound it to a reasonable maximum if it's too large to fit in VRAM
        if width * height > 1024 * 576:
            log.warning("Requested resolution %dx%d is very large for SVD. Clamping to max ~1024x576 area.", width, height)
            width, height = 1024, 576

        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
        except Exception as exc:  # noqa: BLE001
            return self._fail(self.name, f"Failed to open/resize image: {exc}")

        fps = int(scene.get("fps", 16))
        seed = scene.get("seed")
        
        # Map generic motion_strength (0.0-1.0) to SVD's motion_bucket_id (1-255)
        motion_bucket_id = scene.get("motion_bucket_id")
        if motion_bucket_id is None:
            strength = float(scene.get("motion_strength", 0.5))
            motion_bucket_id = max(1, min(255, int(strength * 255)))
        else:
            motion_bucket_id = int(motion_bucket_id)
        # SVD XT generates exactly 25 frames
        frames_out = 25 
        
        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            seed = np.random.randint(0, 2**31)
            generator = torch.manual_seed(seed)

        log.info("SVD generating: %s | %dx%d | 25 frames | motion=%d | seed=%d",
                 image_path.name, width, height, motion_bucket_id, seed)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = self.pipe(
                img,
                decode_chunk_size=8,
                generator=generator,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=0.02,
                num_frames=frames_out
            )
            frames = result.frames[0]
            
            export_to_video(frames, str(output_path), fps=fps)
        except Exception as exc:  # noqa: BLE001
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return self._fail(self.name, f"SVD pipeline generation failed: {exc}")

        generation_time = time.monotonic() - t_start
        
        # Cleanup aggressively after each generation
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        if not output_path.exists():
            return self._fail(self.name, "SVD export_to_video did not create the file.")

        log.info("SVD finished in %.1fs → %s", generation_time, output_path.name)

        return GenerationResult(
            success=True,
            output_path=str(output_path),
            model_used=self.name,
            duration_seconds=frames_out / fps,
            fps=fps,
            width=width,
            height=height,
            seed=seed,
            frame_count=frames_out,
            generation_time_seconds=generation_time,
        )

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._clear_gpu_cache()
            log.debug("SVDAdapter unloaded.")

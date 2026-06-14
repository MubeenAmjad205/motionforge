"""
MotionForge - model_adapters/base_adapter.py
Abstract base class all video generation adapters must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..shared.types import GenerationResult


class BaseAdapter(ABC):
    """
    Contract every model adapter must satisfy.

    Lifecycle:
      1. is_available() — check if model can run in this environment
      2. load()         — load weights / initialise pipeline
      3. generate()     — produce a video and return a GenerationResult
      4. unload()       — release GPU memory / resources

    Rules:
      - generate() MUST NOT raise exceptions for normal failures.
        Return GenerationResult(success=False, error_message=...) instead.
      - generate() MAY raise NotImplementedError if not yet implemented.
        The FallbackManager handles this gracefully.
      - is_available() MUST return False if the adapter is not implemented,
        preventing the pipeline from wasting a retry slot.
    """

    # Unique identifier — must match the key in models.json
    name: str = "base"

    # Set to True only in fully implemented adapters
    IMPLEMENTED: bool = True

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True ONLY if:
          1. The adapter is fully implemented (IMPLEMENTED = True)
          2. All required libraries are importable
          3. The runtime environment is capable (e.g. sufficient VRAM)
        """

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. No-op for CPU-only adapters."""

    @abstractmethod
    def generate(self, scene: dict[str, Any], output_path: Path) -> GenerationResult:
        """
        Generate a motion video for the given scene config.

        Parameters
        ----------
        scene       : Fully-resolved scene dict (merged with global defaults).
        output_path : Destination path for the MP4 file.

        Returns
        -------
        GenerationResult — always returned, never raises on generation failure.
        Raises NotImplementedError if the adapter is not yet implemented.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory. Free GPU cache if applicable."""

    # ---------------------------------------------------------------------- #
    # Helpers available to all adapters
    # ---------------------------------------------------------------------- #

    def _clear_gpu_cache(self) -> None:
        """Attempt to free CUDA cache; silently ignored if no GPU."""
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _resolution_from_scene(self, scene: dict[str, Any]) -> tuple[int, int]:
        """Extract (width, height) from scene dict with safe defaults."""
        res = scene.get("resolution", {})
        return int(res.get("width", 854)), int(res.get("height", 480))

    def _fail(self, model_name: str, error: str) -> GenerationResult:
        """Convenience factory for a failed GenerationResult."""
        return GenerationResult(
            success=False,
            output_path=None,
            model_used=model_name,
            error_message=error,
        )

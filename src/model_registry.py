"""
MotionForge - model_registry.py
Central registry that maps model names to adapter classes.
Loads capabilities from configs/models.json.
"""

import json
from pathlib import Path
from typing import Any

from .logger import get_logger
from .model_adapters.base_adapter import BaseAdapter

log = get_logger("motionforge.model_registry")


class ModelRegistry:
    """
    Holds model capability metadata and provides adapter class lookup.
    New adapters are registered via register().
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, dict[str, Any]] = {}
        self._adapters: dict[str, type[BaseAdapter]] = {}

    # ---------------------------------------------------------------------- #
    # Bootstrap
    # ---------------------------------------------------------------------- #

    def load_capabilities(self, models_json_path: Path) -> None:
        """Parse configs/models.json and store model capability dicts."""
        if not models_json_path.exists():
            raise FileNotFoundError(f"models.json not found: {models_json_path}")
        with models_json_path.open("r", encoding="utf-8") as f:
            self._capabilities = json.load(f)
        log.info("Loaded capabilities for %d model(s).", len(self._capabilities))

    def register(self, model_name: str, adapter_cls: type[BaseAdapter]) -> None:
        """Bind a model name to its adapter implementation class."""
        self._adapters[model_name] = adapter_cls
        log.debug("Registered adapter '%s' → %s", model_name, adapter_cls.__name__)

    # ---------------------------------------------------------------------- #
    # Queries
    # ---------------------------------------------------------------------- #

    def get_capabilities(self, model_name: str) -> dict[str, Any]:
        """Return capability dict for a model, or empty dict if unknown."""
        return self._capabilities.get(model_name, {})

    def get_adapter_class(self, model_name: str) -> type[BaseAdapter] | None:
        """Return the registered adapter class for a model name."""
        return self._adapters.get(model_name)

    def is_registered(self, model_name: str) -> bool:
        return model_name in self._adapters

    def list_models(self) -> list[str]:
        return list(self._adapters.keys())

    # ---------------------------------------------------------------------- #
    # Frame count resolution
    # ---------------------------------------------------------------------- #

    def resolve_frame_count(self, model_name: str, duration_seconds: float, fps: int) -> int:
        """
        Clamp and align requested frames to the model's supported range.
        Falls back to raw calculation if no capabilities are found.
        """
        caps = self.get_capabilities(model_name)
        requested = int(duration_seconds * fps)

        if not caps:
            return max(1, requested)

        min_f = caps.get("min_frames", 1)
        max_f = caps.get("max_frames", 9999)
        multiple = caps.get("frame_multiple", 1)

        frames = max(min_f, min(requested, max_f))
        if multiple > 1:
            frames = frames - (frames % multiple)
            frames = max(min_f, frames)

        return frames

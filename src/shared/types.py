"""
MotionForge - shared/types.py
Shared type definitions used across the entire pipeline.
Single source of truth for all data contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationResult:
    """
    Structured result returned by every adapter's generate() method.
    The FallbackManager consumes and propagates this object.
    """

    # Core outcome
    success: bool
    output_path: Optional[str]       # Absolute path to MP4, or None on failure
    model_used: str                   # Name key matching models.json

    # Video properties (filled on success)
    duration_seconds: Optional[float] = None
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    frame_count: Optional[int] = None

    # Timing
    generation_time_seconds: float = 0.0

    # Error info (filled on failure)
    error_message: Optional[str] = None
    is_not_implemented: bool = False   # True when adapter raised NotImplementedError

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "model_used": self.model_used,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "frame_count": self.frame_count,
            "generation_time_seconds": round(self.generation_time_seconds, 3),
            "error_message": self.error_message,
        }


@dataclass
class SceneResult:
    """
    Full result record for one scene after pipeline processing.
    Written to outputs/logs/scene_NNN.json and used in reports.
    """

    scene_id: int
    scene_name: str
    status: str                        # pending | running | success | failed | skipped
    model_used: Optional[str] = None
    output_path: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    generation_time_seconds: float = 0.0

    # Video metadata (filled on success)
    duration_seconds: Optional[float] = None
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    frame_count: Optional[int] = None
    file_size_bytes: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "scene_name": self.scene_name,
            "status": self.status,
            "model_used": self.model_used,
            "output_path": self.output_path,
            "attempts": self.attempts,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "generation_time_seconds": round(self.generation_time_seconds, 3),
            "video": {
                "duration_seconds": self.duration_seconds,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "seed": self.seed,
                "frame_count": self.frame_count,
                "file_size_bytes": self.file_size_bytes,
            } if self.duration_seconds is not None else None,
        }

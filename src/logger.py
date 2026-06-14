"""
MotionForge - logger.py
Centralised logging for global pipeline and per-scene files.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Module-level logger used by the pipeline entry point
# --------------------------------------------------------------------------- #
_pipeline_logger: Optional[logging.Logger] = None


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_pipeline_logger(logs_dir: Path) -> logging.Logger:
    """
    Create (or return cached) global pipeline logger.
    Writes to logs/project.log and stderr simultaneously.
    """
    global _pipeline_logger
    if _pipeline_logger is not None:
        return _pipeline_logger

    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("motionforge")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = _build_formatter()

    # File handler — full detail
    fh = logging.FileHandler(logs_dir / "project.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Error-only file handler
    eh = logging.FileHandler(logs_dir / "errors.log", encoding="utf-8")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(formatter)
    logger.addHandler(eh)

    # Stderr handler for Colab visibility
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    _pipeline_logger = logger
    return logger


def get_logger(name: str = "motionforge") -> logging.Logger:
    """Return a child logger under the motionforge namespace."""
    return logging.getLogger(name)


def build_scene_logger(scene_name: str, log_path: Path) -> logging.Logger:
    """
    Create a file-only logger for a single scene.
    Writes full structured log to outputs/<scene>/scene_log.txt.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    name = f"motionforge.scene.{scene_name}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = True  # bubble up to global logger

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_build_formatter())
    logger.addHandler(fh)

    return logger


def log_scene_header(logger: logging.Logger, scene: dict) -> None:
    """Write a structured header block to a scene log."""
    logger.info("=" * 60)
    logger.info("SCENE START")
    logger.info("ID        : %s", scene.get("id"))
    logger.info("Name      : %s", scene.get("name"))
    logger.info("Model     : %s", scene.get("_resolved_model", "—"))
    logger.info("Image     : %s", scene.get("input_image"))
    logger.info("Duration  : %ss", scene.get("duration_seconds"))
    logger.info("FPS       : %s", scene.get("fps"))
    logger.info("Resolution: %sx%s", scene.get("resolution", {}).get("width"), scene.get("resolution", {}).get("height"))
    logger.info("Seed      : %s", scene.get("seed"))
    logger.info("Prompt    : %s", scene.get("motion_prompt", "")[:120])
    logger.info("Timestamp : %s", datetime.utcnow().isoformat())
    logger.info("=" * 60)


def log_scene_footer(logger: logging.Logger, status: str, output_path: Optional[str], error: Optional[str]) -> None:
    """Write a structured footer block to a scene log."""
    logger.info("-" * 60)
    logger.info("SCENE END")
    logger.info("Status    : %s", status)
    logger.info("Output    : %s", output_path or "—")
    if error:
        logger.error("Error     : %s", error)
    logger.info("Timestamp : %s", datetime.utcnow().isoformat())
    logger.info("-" * 60)

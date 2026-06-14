"""
MotionForge - zip_exporter.py
Packages all pipeline outputs into a single downloadable ZIP.

ZIP structure:
  clips/
    scene_001.mp4
    scene_002.mp4
    scene_001.png          (input image copies)
  logs/
    scene_001.json
    scene_002.json
    scene_001_pipeline.txt (raw pipeline log)
  reports/
    scene_001.md
    scene_002.md
    generation_report.md
    project_manifest.json
    failed_scenes.json
  config/
    models.json
    default_settings.json
  input/
    scenes.json
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from .logger import get_logger

log = get_logger("motionforge.zip_exporter")


def build_zip(
    output_root: Path,
    zip_path: Path,
    results: list[dict[str, Any]],
    scenes_json_path: Path | None = None,
    configs_dir: Path | None = None,
    extra_files: list[Path] | None = None,
) -> Path:
    """
    Build the output ZIP with the full spec-compliant directory structure.

    Parameters
    ----------
    output_root    : Root outputs directory (contains clips/, logs/, reports/).
    zip_path       : Destination for the ZIP file.
    results        : Result records from QueueManager.process_all().
    scenes_json_path : Optional path to original scenes.json.
    configs_dir    : Optional path to configs/ for inclusion.
    extra_files    : Optional additional files to include at ZIP root.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Building ZIP → %s", zip_path)

    clips_dir = output_root / "clips"
    logs_dir = output_root / "logs"
    reports_dir = output_root / "reports"

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:

        # clips/ — MP4 files and input image copies
        if clips_dir.exists():
            for f in sorted(clips_dir.iterdir()):
                if f.is_file():
                    zf.write(str(f), f"clips/{f.name}")

        # logs/ — per-scene JSON logs and pipeline txt logs
        if logs_dir.exists():
            for f in sorted(logs_dir.iterdir()):
                if f.is_file():
                    zf.write(str(f), f"logs/{f.name}")

        # reports/ — per-scene Markdown files
        if reports_dir.exists():
            for f in sorted(reports_dir.iterdir()):
                if f.is_file():
                    zf.write(str(f), f"reports/{f.name}")

        # reports/ — global reports at root of outputs/
        for report_name in ("project_manifest.json", "generation_report.md", "failed_scenes.json"):
            report_path = output_root / report_name
            if report_path.exists():
                zf.write(str(report_path), f"reports/{report_name}")

        # config/ — model and settings configs
        if configs_dir and configs_dir.exists():
            for f in sorted(configs_dir.iterdir()):
                if f.is_file() and f.suffix == ".json":
                    zf.write(str(f), f"config/{f.name}")

        # input/ — original scenes.json
        if scenes_json_path and scenes_json_path.exists():
            zf.write(str(scenes_json_path), f"input/{scenes_json_path.name}")

        # extra files at zip root
        for extra in (extra_files or []):
            if extra.exists():
                zf.write(str(extra), extra.name)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    log.info("ZIP created: %s (%.2f MB)", zip_path, size_mb)
    return zip_path

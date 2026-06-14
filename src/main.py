"""
MotionForge - src/main.py
Pipeline entry point. Wires all modules and runs the full pipeline.

Usage (from motionforge/ project root):
    python src/main.py
    python src/main.py --scenes data/scenes.json
    python src/main.py --scenes data/scenes.json --resume
    python src/main.py --adapter mock_adapter    (force a specific adapter)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap: project root is parent of src/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import setup_pipeline_logger, get_logger
from src.scene_loader import load_project, load_scenes
from src.validators import validate_all_scenes
from src.model_registry import ModelRegistry
from src.model_adapters import (
    MockAdapter,
    ImagePanZoomAdapter,
    SVDAdapter,
    WanAdapter,
    FramePackAdapter,
)
from src.fallback_manager import FallbackManager
from src.queue_manager import QueueManager
from src.report_generator import (
    build_manifest,
    save_manifest,
    save_failed_scenes,
    build_markdown_report,
    save_markdown_report,
)
from src.zip_exporter import build_zip

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
ZIP_DIR = PROJECT_ROOT / "zip"

DEFAULT_SCENES_PATH = DATA_DIR / "scenes.json"
MODELS_JSON_PATH = CONFIGS_DIR / "models.json"
CHECKPOINT_PATH = OUTPUT_DIR / "project_queue_state.json"


# --------------------------------------------------------------------------- #
# Registry bootstrap
# --------------------------------------------------------------------------- #

def build_registry() -> ModelRegistry:
    """Create and populate the model registry with all known adapters."""
    registry = ModelRegistry()
    registry.load_capabilities(MODELS_JSON_PATH)

    # Register adapters — order here does not affect fallback priority
    # (that is controlled by global_settings.fallback_models in scenes.json)
    registry.register("image_pan_zoom", ImagePanZoomAdapter)
    registry.register("mock_adapter", MockAdapter)
    registry.register("stable_video_diffusion_xt", SVDAdapter)
    registry.register("wan2_1_i2v_gguf_q4", WanAdapter)
    registry.register("framepack_low_vram", FramePackAdapter)

    return registry


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def run_pipeline(
    scenes_path: Path,
    resume: bool = False,
    adapter_override: str | None = None,
) -> None:
    """
    Execute the full MotionForge pipeline.

    Parameters
    ----------
    scenes_path      : Path to scenes.json.
    resume           : If True, load checkpoint and skip completed scenes.
    adapter_override : Force all scenes to use this model name.
    """
    log = get_logger("motionforge.main")

    separator = "=" * 60
    log.info(separator)
    log.info("MotionForge Pipeline Starting")
    log.info("Scenes     : %s", scenes_path)
    log.info("Output dir : %s", OUTPUT_DIR)
    if adapter_override:
        log.info("Adapter    : %s (forced override)", adapter_override)
    log.info(separator)

    # 1. Load and parse project JSON
    project = load_project(scenes_path)
    global_settings = project.get("global_settings", {})
    queue_settings = project.get("queue_settings", {})

    # 2. Build registry first so we can validate model names
    registry = build_registry()

    # 3. Load and validate scenes (pass registered models for model_override check)
    raw_scenes = load_scenes(project)
    valid_scenes = validate_all_scenes(
        raw_scenes,
        PROJECT_ROOT,
        registered_models=registry.list_models(),
    )

    if not valid_scenes:
        log.error("No valid scenes found — aborting.")
        sys.exit(1)

    # Convert relative image paths to absolute for the adapters
    for scene in valid_scenes:
        if "input_image" in scene:
            scene["input_image"] = str(PROJECT_ROOT / scene["input_image"])

    # Apply adapter override if requested
    if adapter_override:
        if not registry.is_registered(adapter_override):
            log.error("Unknown adapter override '%s'. Registered: %s", adapter_override, registry.list_models())
            sys.exit(1)
        for scene in valid_scenes:
            scene["_resolved_model"] = adapter_override
        log.info("Adapter override applied: all scenes will use '%s'.", adapter_override)

    # 4. Configure fallback chain
    fallback_models: list[str] = global_settings.get("fallback_models", ["mock_adapter"])
    max_retries: int = queue_settings.get("max_retries_per_scene", 2)

    fallback_manager = FallbackManager(
        registry=registry,
        fallback_models=fallback_models,
        max_retries=max_retries,
    )

    # 5. Build and optionally resume queue
    queue_mgr = QueueManager(
        output_root=OUTPUT_DIR,
        fallback_manager=fallback_manager,
        project_root=PROJECT_ROOT,
        continue_on_error=queue_settings.get("continue_on_error", True),
        checkpoint_after_each=queue_settings.get("checkpoint_after_each_scene", True),
    )
    queue_mgr.build_queue(valid_scenes)

    if resume and CHECKPOINT_PATH.exists():
        loaded = queue_mgr.load_checkpoint(CHECKPOINT_PATH)
        log.info("Resume mode: checkpoint %s.", "loaded" if loaded else "not found")

    # 6. Process all scenes
    results = queue_mgr.process_all(CHECKPOINT_PATH)

    # 7. Generate global reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(project, results, zip_path=None)
    save_manifest(manifest, OUTPUT_DIR)
    save_failed_scenes(results, OUTPUT_DIR)
    report_md = build_markdown_report(manifest)
    save_markdown_report(report_md, OUTPUT_DIR)

    # 8. Package ZIP
    project_name = project.get("project", {}).get("name", "motionforge_output")
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ZIP_DIR / f"{project_name}_output.zip"

    build_zip(
        output_root=OUTPUT_DIR,
        zip_path=zip_path,
        results=results,
        scenes_json_path=scenes_path,
        configs_dir=CONFIGS_DIR,
    )

    # Update manifest with final zip path
    manifest["output_zip"] = str(zip_path)
    save_manifest(manifest, OUTPUT_DIR)

    # 9. Print summary
    summary = queue_mgr.get_summary()
    log.info(separator)
    log.info("Pipeline Complete")
    log.info("  Successful : %d", summary.get("success", 0))
    log.info("  Failed     : %d", summary.get("failed", 0))
    log.info("  ZIP        : %s", zip_path)
    log.info(separator)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MotionForge — Image-to-Motion Video Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/main.py\n"
            "  python src/main.py --scenes data/scenes.json\n"
            "  python src/main.py --adapter image_pan_zoom\n"
            "  python src/main.py --resume\n"
        ),
    )
    parser.add_argument(
        "--scenes", type=Path, default=DEFAULT_SCENES_PATH,
        help="Path to scenes JSON (default: data/scenes.json)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint (skip already-completed scenes)",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Force all scenes to use this model adapter name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    setup_pipeline_logger(LOGS_DIR)
    run_pipeline(
        scenes_path=args.scenes,
        resume=args.resume,
        adapter_override=args.adapter,
    )

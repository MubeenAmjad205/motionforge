"""
MotionForge - fallback_manager.py
Orchestrates the primary → retry → fallback model chain for each scene.
Reads model order from scene config and global fallback_models list.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .logger import get_logger, log_scene_header, log_scene_footer
from .model_registry import ModelRegistry
from .validators import validate_video_output
from .shared.types import GenerationResult

log = get_logger("motionforge.fallback_manager")

# Seconds to wait between retry attempts
_RETRY_DELAY_SECONDS = 2


class FallbackManager:
    """
    Manages the full generation lifecycle for one scene:
      1. Try primary model (from scene._resolved_model)
      2. Retry same model up to max_retries on failure
      3. Try each fallback model in order if primary exhausted
      4. Skip models that are not registered or not available
      5. Mark scene failed only when all options are exhausted
      6. Always log attempts and errors — never silently swallow failures
    """

    def __init__(
        self,
        registry: ModelRegistry,
        fallback_models: list[str],
        max_retries: int = 2,
    ) -> None:
        self._registry = registry
        self._fallback_models = fallback_models
        self._max_retries = max_retries

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def run_scene(
        self,
        scene: dict[str, Any],
        output_path: Path,
        scene_logger: Any,
    ) -> GenerationResult:
        """
        Attempt video generation with primary then fallback models.

        Returns a GenerationResult. Never raises.
        """
        primary_model = scene.get("_resolved_model", "image_pan_zoom")
        # Build model chain: primary first, then fallbacks (excluding primary)
        model_chain = [primary_model] + [
            m for m in self._fallback_models if m != primary_model
        ]

        log_scene_header(scene_logger, scene)

        total_attempts = 0
        last_error: str | None = None

        for model_name in model_chain:
            if not self._registry.is_registered(model_name):
                scene_logger.warning("Model '%s' not registered — skipping.", model_name)
                continue

            adapter_cls = self._registry.get_adapter_class(model_name)
            if adapter_cls is None:
                continue

            adapter = adapter_cls()

            if not adapter.is_available():
                scene_logger.info(
                    "Model '%s' not available (IMPLEMENTED=%s) — skipping.",
                    model_name,
                    getattr(adapter_cls, "IMPLEMENTED", "?"),
                )
                continue

            # Retry loop for this model
            for attempt in range(1, self._max_retries + 2):  # +2: initial + N retries
                total_attempts += 1
                scene_logger.info(
                    "Attempt %d/%d — model: %s",
                    attempt,
                    self._max_retries + 1,
                    model_name,
                )

                t_attempt_start = time.monotonic()

                try:
                    adapter.load()
                    result: GenerationResult = adapter.generate(scene, output_path)
                    adapter.unload()

                except NotImplementedError as exc:
                    last_error = str(exc)
                    scene_logger.warning(
                        "Model '%s' raised NotImplementedError (skipping): %s", model_name, last_error
                    )
                    try:
                        adapter.unload()
                    except Exception:  # noqa: BLE001
                        pass
                    break  # No retries for unimplemented adapters

                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    scene_logger.error(
                        "Attempt %d raised exception for model '%s': %s",
                        attempt, model_name, last_error,
                    )
                    try:
                        adapter.unload()
                    except Exception:  # noqa: BLE001
                        pass
                    self._clear_cuda()

                    if attempt <= self._max_retries:
                        scene_logger.info("Retrying in %ds...", _RETRY_DELAY_SECONDS)
                        time.sleep(_RETRY_DELAY_SECONDS)
                    else:
                        scene_logger.warning(
                            "All %d attempt(s) for '%s' failed. Trying next model.",
                            self._max_retries + 1, model_name,
                        )
                    continue

                else:
                    # generate() returned a result — check for success/failure
                    if not result.success:
                        last_error = result.error_message or "Adapter returned failure"
                        scene_logger.error(
                            "Attempt %d failed (adapter reported failure) for '%s': %s",
                            attempt, model_name, last_error,
                        )
                        self._clear_cuda()

                        if attempt <= self._max_retries:
                            scene_logger.info("Retrying in %ds...", _RETRY_DELAY_SECONDS)
                            time.sleep(_RETRY_DELAY_SECONDS)
                        else:
                            scene_logger.warning(
                                "All retries exhausted for '%s'. Trying next model.", model_name
                            )
                        continue

                    # Validate the output file
                    output_file = Path(result.output_path)
                    ok, msg = validate_video_output(
                        output_file,
                        expected_duration=float(scene.get("duration_seconds", 4)),
                    )
                    if not ok:
                        last_error = f"Output validation failed: {msg}"
                        scene_logger.error("Attempt %d: %s", attempt, last_error)
                        self._clear_cuda()

                        if attempt <= self._max_retries:
                            scene_logger.info("Retrying in %ds...", _RETRY_DELAY_SECONDS)
                            time.sleep(_RETRY_DELAY_SECONDS)
                        else:
                            scene_logger.warning(
                                "All retries exhausted for '%s'. Trying next model.", model_name
                            )
                        continue

                    # ── SUCCESS ────────────────────────────────────────── #
                    scene_logger.info(
                        "Generation succeeded: model='%s' output='%s' time=%.1fs",
                        model_name, result.output_path, result.generation_time_seconds,
                    )
                    log_scene_footer(scene_logger, "success", result.output_path, None)
                    return result

        # All models and retries exhausted
        log.error(
            "Scene '%s' failed after %d total attempt(s). Last error: %s",
            scene.get("name"), total_attempts, last_error,
        )
        log_scene_footer(scene_logger, "failed", None, last_error)

        return GenerationResult(
            success=False,
            output_path=None,
            model_used="none",
            error_message=last_error or "All models failed or unavailable",
        )

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _clear_cuda() -> None:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

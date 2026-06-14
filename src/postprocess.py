"""
MotionForge - postprocess.py
Post-processing utilities using MoviePy.
Handles FPS normalisation, re-encoding, trimming, and padding.
"""

from pathlib import Path
from typing import Any

from .logger import get_logger

log = get_logger("motionforge.postprocess")


def normalise_fps(input_path: Path, output_path: Path, target_fps: int) -> Path:
    """Re-encode a video to exactly target_fps."""
    from moviepy.editor import VideoFileClip  # type: ignore

    clip = VideoFileClip(str(input_path))
    clip = clip.set_fps(target_fps)
    clip.write_videofile(str(output_path), codec="libx264", audio=False, logger=None)
    clip.close()
    log.debug("FPS normalised → %s (%d fps)", output_path, target_fps)
    return output_path


def trim_clip(input_path: Path, output_path: Path, duration: float) -> Path:
    """Trim a video to exactly `duration` seconds."""
    from moviepy.editor import VideoFileClip  # type: ignore

    clip = VideoFileClip(str(input_path))
    if clip.duration > duration:
        clip = clip.subclip(0, duration)
    clip.write_videofile(str(output_path), codec="libx264", audio=False, logger=None)
    clip.close()
    log.debug("Trimmed to %.2fs → %s", duration, output_path)
    return output_path


def pad_or_loop_clip(input_path: Path, output_path: Path, target_duration: float) -> Path:
    """
    Loop the clip if shorter than target_duration,
    then trim to exactly target_duration.
    """
    from moviepy.editor import VideoFileClip, concatenate_videoclips  # type: ignore

    clip = VideoFileClip(str(input_path))
    if clip.duration >= target_duration:
        result = clip.subclip(0, target_duration)
    else:
        repeats = int(target_duration / clip.duration) + 2
        looped = concatenate_videoclips([clip] * repeats)
        result = looped.subclip(0, target_duration)

    result.write_videofile(str(output_path), codec="libx264", audio=False, logger=None)
    result.close()
    clip.close()
    log.debug("Padded/looped to %.2fs → %s", target_duration, output_path)
    return output_path


def extract_thumbnail(video_path: Path, thumb_path: Path, time_sec: float = 0.5) -> Path:
    """Save a single frame from the video as a JPEG thumbnail."""
    from moviepy.editor import VideoFileClip  # type: ignore

    clip = VideoFileClip(str(video_path))
    t = min(time_sec, clip.duration - 0.01)
    frame = clip.get_frame(t)
    clip.close()

    from PIL import Image  # type: ignore
    import numpy as np

    img = Image.fromarray(frame.astype(np.uint8))
    img.save(str(thumb_path), "JPEG", quality=85)
    log.debug("Thumbnail saved → %s", thumb_path)
    return thumb_path


def get_video_info(video_path: Path) -> dict[str, Any]:
    """Return basic metadata dict for a video file."""
    try:
        from moviepy.editor import VideoFileClip  # type: ignore

        clip = VideoFileClip(str(video_path))
        info = {
            "duration": round(clip.duration, 3),
            "fps": clip.fps,
            "width": clip.w,
            "height": clip.h,
            "file_size_bytes": video_path.stat().st_size,
        }
        clip.close()
        return info
    except Exception as exc:  # noqa: BLE001
        log.error("Could not read video info for %s: %s", video_path, exc)
        return {}

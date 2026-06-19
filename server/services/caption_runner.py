"""Wrapper around auto_caption.py for the web training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Import from bundled scripts directory
from server.scripts.auto_caption import generate_captions


def run(config: dict[str, Any], callback: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
    """Run the auto-captioning pipeline.

    Parameters
    ----------
    config : dict
        Full merged training configuration (must contain ``data`` and ``caption`` keys).
    callback : callable
        ``callback(event_type, data)`` called at start and completion.

    Returns
    -------
    dict
        ``{"captioned": int, "failed": int}``
    """
    callback("caption_start", {"message": "Starting auto-captioning"})

    try:
        # Run the captioning
        generate_captions(config)

        # Count results
        processed_dir = Path(config["data"]["processed_dir"])
        caption_files = [
            f for f in processed_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        ] if processed_dir.exists() else []

        captioned_count = len(caption_files)

        # Count images without captions
        image_files = [
            f for f in processed_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ] if processed_dir.exists() else []

        failed_count = max(0, len(image_files) - captioned_count)

        result = {"captioned": captioned_count, "failed": failed_count}
        callback("caption_complete", result)
        return result

    except Exception as exc:
        logger.exception("Captioning failed")
        callback("caption_complete", {"captioned": 0, "failed": 0, "error": str(exc)})
        raise

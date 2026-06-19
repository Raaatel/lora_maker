"""Wrapper around preprocess.py for the web training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Import from bundled scripts directory
from server.scripts.preprocess import process_images


def run(config: dict[str, Any], callback: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
    """Run the preprocessing pipeline.

    Parameters
    ----------
    config : dict
        Full merged training configuration (must contain ``data`` and ``preprocess`` keys).
    callback : callable
        ``callback(event_type, data)`` called at start and completion.

    Returns
    -------
    dict
        ``{"processed": int, "failed": int}``
    """
    callback("preprocess_start", {"message": "Starting image preprocessing"})

    try:
        # Count files before processing
        raw_dir = Path(config["data"]["raw_dir"])
        supported = set(config.get("preprocess", {}).get(
            "supported_formats", [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
        ))
        image_files = [
            f for f in raw_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        ] if raw_dir.exists() else []

        total_before = len(image_files)

        # Run the preprocessing
        process_images(config)

        # Count results
        processed_dir = Path(config["data"]["processed_dir"])
        processed_images = [
            f for f in processed_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ] if processed_dir.exists() else []

        processed_count = len(processed_images)
        failed_count = max(0, total_before - processed_count)

        result = {"processed": processed_count, "failed": failed_count}
        callback("preprocess_complete", result)
        return result

    except Exception as exc:
        logger.exception("Preprocessing failed")
        callback("preprocess_complete", {"processed": 0, "failed": 0, "error": str(exc)})
        raise

"""
01_preprocess.py - Webtoon character face/hair image preprocessing for SDXL LoRA training.

Detects anime-style faces, crops with margin (extra top margin for hair),
resizes to target resolution, and saves to processed directory.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


def imread_unicode(filepath: str) -> np.ndarray:
    """Read image with unicode path support (handles Korean filenames)."""
    buf = np.fromfile(filepath, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(filepath: str, image: np.ndarray) -> bool:
    """Write image with unicode path support (handles Korean filenames)."""
    ext = Path(filepath).suffix
    result, buf = cv2.imencode(ext, image)
    if result:
        buf.tofile(filepath)
        return True
    return False


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_face_detector():
    """Load anime/cartoon face detector. Falls back to default Haar cascade."""
    # Try lbpcascade_animeface first (optimized for anime/webtoon)
    anime_cascade_path = Path(cv2.data.haarcascades).parent / "lbpcascade_animeface.xml"
    if anime_cascade_path.exists():
        detector = cv2.CascadeClassifier(str(anime_cascade_path))
        if not detector.empty():
            print("[INFO] Using lbpcascade_animeface detector")
            return detector

    # Try downloading anime face cascade
    try:
        import urllib.request

        url = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
        cache_path = Path(__file__).parent / "lbpcascade_animeface.xml"
        if not cache_path.exists():
            print("[INFO] Downloading anime face detector...")
            urllib.request.urlretrieve(url, str(cache_path))
        detector = cv2.CascadeClassifier(str(cache_path))
        if not detector.empty():
            print("[INFO] Using downloaded lbpcascade_animeface detector")
            return detector
    except Exception as e:
        print(f"[WARN] Failed to download anime face detector: {e}")

    # Fallback to default Haar cascade
    print("[INFO] Falling back to default Haar cascade face detector")
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        print("[ERROR] Failed to load any face detector")
        sys.exit(1)
    return detector


def detect_faces(image: np.ndarray, detector: cv2.CascadeClassifier) -> list:
    """Detect faces in image, returns list of (x, y, w, h)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if isinstance(faces, tuple):
        return []
    return faces.tolist()


def crop_face_with_margin(
    image: np.ndarray,
    face_rect: list,
    margin_top: float = 2.0,
    margin_sides: float = 1.5,
) -> np.ndarray:
    """Crop face region with asymmetric margins (more on top for hair)."""
    x, y, w, h = face_rect
    img_h, img_w = image.shape[:2]

    # Calculate margins
    margin_left = int(w * (margin_sides - 1) / 2)
    margin_right = int(w * (margin_sides - 1) / 2)
    margin_top_px = int(h * (margin_top - 1))
    margin_bottom = int(h * (margin_sides - 1) / 2)

    # Calculate crop coordinates with bounds checking
    x1 = max(0, x - margin_left)
    y1 = max(0, y - margin_top_px)
    x2 = min(img_w, x + w + margin_right)
    y2 = min(img_h, y + h + margin_bottom)

    return image[y1:y2, x1:x2]


def resize_image(
    image: np.ndarray,
    target_size: int,
    mode: str = "crop",
    pad_color: list = None,
) -> np.ndarray:
    """Resize image to target_size x target_size."""
    h, w = image.shape[:2]

    if mode == "crop":
        # Resize so shorter side = target_size, then center crop
        scale = target_size / min(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Center crop
        start_x = (new_w - target_size) // 2
        start_y = (new_h - target_size) // 2
        image = image[start_y : start_y + target_size, start_x : start_x + target_size]

    elif mode == "pad":
        # Resize so longer side = target_size, then pad
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Pad to square
        pad_color = pad_color or [255, 255, 255]
        canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
        start_x = (target_size - new_w) // 2
        start_y = (target_size - new_h) // 2
        canvas[start_y : start_y + new_h, start_x : start_x + new_w] = image
        image = canvas

    return image


def process_images(config: dict):
    """Main preprocessing pipeline."""
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    preprocess_cfg = config["preprocess"]

    processed_dir.mkdir(parents=True, exist_ok=True)

    supported = set(preprocess_cfg["supported_formats"])
    target_size = preprocess_cfg["target_size"]
    margin_top = preprocess_cfg["face_margin_top"]
    margin_sides = preprocess_cfg["face_margin_sides"]
    resize_mode = preprocess_cfg["resize_mode"]
    pad_color = preprocess_cfg.get("pad_color", [255, 255, 255])

    # Collect image files
    image_files = sorted(
        f
        for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported
    )

    if not image_files:
        print(f"[ERROR] No images found in {raw_dir}")
        print(f"        Supported formats: {supported}")
        sys.exit(1)

    print(f"[INFO] Found {len(image_files)} images in {raw_dir}")

    detector = get_face_detector()

    total = 0
    success = 0
    failed = 0
    no_face = 0
    multi_face = 0

    for img_path in tqdm(image_files, desc="Processing"):
        try:
            image = imread_unicode(str(img_path))
            if image is None:
                print(f"[WARN] Cannot read: {img_path.name}")
                failed += 1
                continue

            total += 1
            faces = detect_faces(image, detector)

            if len(faces) == 0:
                # No face detected - use entire image
                print(f"[WARN] No face detected in {img_path.name}, using full image")
                no_face += 1
                cropped = image
            else:
                if len(faces) > 1:
                    # Multiple faces - use largest
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    multi_face += 1

                cropped = crop_face_with_margin(
                    image, faces[0], margin_top, margin_sides
                )

            # Resize to target
            result = resize_image(cropped, target_size, resize_mode, pad_color)

            # Save with ascii-safe name to avoid encoding issues
            out_name = f"image_{total:03d}.png"
            out_path = processed_dir / out_name
            if imwrite_unicode(str(out_path), result):
                success += 1
            else:
                print(f"[ERROR] Failed to save: {out_path}")
                failed += 1
                continue

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path.name}: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 50)
    print("Preprocessing Summary")
    print("=" * 50)
    print(f"  Total images found:      {len(image_files)}")
    print(f"  Successfully processed:  {success}")
    print(f"  No face detected (used full image): {no_face}")
    print(f"  Multiple faces (used largest):      {multi_face}")
    print(f"  Failed:                  {failed}")
    print(f"  Output directory:        {processed_dir}")
    print(f"  Output resolution:       {target_size}x{target_size}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess webtoon images for SDXL LoRA training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Override raw image directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.raw_dir:
        config["data"]["raw_dir"] = args.raw_dir
    if args.output_dir:
        config["data"]["processed_dir"] = args.output_dir

    process_images(config)


if __name__ == "__main__":
    main()

"""
02_auto_caption.py - Auto-captioning for SDXL LoRA training images.

Supports two captioning methods:
  - WD-Tagger: danbooru-style tag-based captions
  - BLIP2: natural language description captions

Prepends trigger word to all captions.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class WDTagger:
    """WD-Tagger (SwinV2-based) for danbooru-style tagging."""

    def __init__(self, model_name: str, threshold: float = 0.35):
        from transformers import pipeline

        self.threshold = threshold
        self.model_name = model_name
        self.pipe = None

    def load(self):
        """Load the model lazily."""
        import timm
        import huggingface_hub
        import pandas as pd

        print(f"[INFO] Loading WD-Tagger model: {self.model_name}")

        # Download model files
        repo_id = self.model_name
        model_path = huggingface_hub.hf_hub_download(repo_id, "model.onnx")
        labels_path = huggingface_hub.hf_hub_download(repo_id, "selected_tags.csv")

        # Load labels
        df = pd.read_csv(labels_path)
        self.tag_names = df["name"].tolist()
        self.general_indices = list(df[df["category"] == 0].index)
        self.character_indices = list(df[df["category"] == 4].index)

        # Load ONNX model
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            print("[INFO] WD-Tagger loaded with ONNX runtime")
        except ImportError:
            print("[ERROR] onnxruntime is required for WD-Tagger.")
            print("        Install with: pip install onnxruntime-gpu")
            sys.exit(1)

    def predict(self, image: Image.Image) -> list[str]:
        """Predict tags for an image."""
        if self.session is None:
            self.load()

        # Preprocess
        size = self.input_shape[1] if self.input_shape[1] else 448
        img = image.convert("RGB").resize((size, size), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        # BGR conversion and normalization
        img_array = img_array[:, :, ::-1]  # RGB to BGR
        img_array = np.expand_dims(img_array, axis=0)

        # Inference
        probs = self.session.run(None, {self.input_name: img_array})[0][0]

        # Filter tags by threshold
        tags = []
        for idx in self.general_indices:
            if probs[idx] >= self.threshold:
                tag = self.tag_names[idx]
                tag = tag.replace("_", " ")
                tags.append((tag, float(probs[idx])))

        # Sort by confidence
        tags.sort(key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in tags]


class BLIP2Captioner:
    """BLIP2-based natural language captioning."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.processor = None
        self.model = None

    def load(self):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        print(f"[INFO] Loading BLIP2 model: {self.model_name}")
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("[INFO] BLIP2 model loaded")

    def predict(self, image: Image.Image) -> str:
        if self.processor is None:
            self.load()

        inputs = self.processor(images=image, return_tensors="pt").to(
            self.model.device, dtype=torch.float16
        )
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        caption = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return caption


def generate_captions(config: dict):
    """Main captioning pipeline."""
    processed_dir = Path(config["data"]["processed_dir"])
    captions_dir = Path(config["data"]["captions_dir"])
    caption_cfg = config["caption"]

    captions_dir.mkdir(parents=True, exist_ok=True)

    method = caption_cfg["method"]
    trigger_word = caption_cfg["trigger_word"]

    # Collect images
    image_files = sorted(
        f
        for f in processed_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )

    if not image_files:
        print(f"[ERROR] No images found in {processed_dir}")
        print("        Run 01_preprocess.py first.")
        sys.exit(1)

    print(f"[INFO] Found {len(image_files)} images")
    print(f"[INFO] Captioning method: {method}")
    print(f"[INFO] Trigger word: {trigger_word}")

    # Initialize captioner
    if method == "wd-tagger":
        captioner = WDTagger(
            model_name=caption_cfg["wd_tagger_model"],
            threshold=caption_cfg.get("wd_tagger_threshold", 0.35),
        )
        captioner.load()
    elif method == "blip2":
        captioner = BLIP2Captioner(model_name=caption_cfg["blip2_model"])
        captioner.load()
    else:
        print(f"[ERROR] Unknown caption method: {method}")
        sys.exit(1)

    success = 0
    failed = 0

    for img_path in tqdm(image_files, desc="Captioning"):
        try:
            image = Image.open(img_path).convert("RGB")

            if method == "wd-tagger":
                tags = captioner.predict(image)
                # Filter out excluded tags (e.g. hair color tags for hair_color LoRA)
                exclude_tags = set(caption_cfg.get("exclude_tags", []))
                if exclude_tags:
                    tags = [t for t in tags if t not in exclude_tags]
                # Build caption: trigger_word, tag1, tag2, ...
                caption = ", ".join([trigger_word] + tags)
            else:
                raw_caption = captioner.predict(image)
                caption = f"{trigger_word}, {raw_caption}"

            # Save caption as .txt with same stem
            caption_path = captions_dir / f"{img_path.stem}.txt"
            caption_path.write_text(caption, encoding="utf-8")

            # Also save next to image for training compatibility
            beside_image_path = processed_dir / f"{img_path.stem}.txt"
            beside_image_path.write_text(caption, encoding="utf-8")

            success += 1

        except Exception as e:
            print(f"[ERROR] Failed to caption {img_path.name}: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 50)
    print("Captioning Summary")
    print("=" * 50)
    print(f"  Total images:     {len(image_files)}")
    print(f"  Captions created: {success}")
    print(f"  Failed:           {failed}")
    print(f"  Method:           {method}")
    print(f"  Trigger word:     {trigger_word}")
    print(f"  Captions dir:     {captions_dir}")
    print("=" * 50)
    print()
    print("[TIP] Review and edit the generated captions before training!")
    print(f"      Caption files are in: {captions_dir}/")
    print("      Each .txt file corresponds to an image with the same name.")
    print("      You can manually add/remove tags to improve training quality.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for SDXL LoRA training images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["wd-tagger", "blip2"],
        default=None,
        help="Override captioning method",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.method:
        config["caption"]["method"] = args.method

    generate_captions(config)


if __name__ == "__main__":
    main()

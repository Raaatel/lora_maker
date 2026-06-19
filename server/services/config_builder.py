"""Config builder with auto num_repeats calculation."""

import copy
import math
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

PRESET_NAMES = {
    "style": "style_lora",
    "character": "character_lora",
    "face": "face_lora",
    "object": "object_lora",
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_defaults() -> Dict[str, Any]:
    with open(CONFIG_DIR / "defaults.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_preset(lora_type: str) -> Dict[str, Any]:
    preset_file = PRESET_NAMES.get(lora_type, f"{lora_type}_lora")
    preset_path = CONFIG_DIR / "presets" / f"{preset_file}.yaml"
    if not preset_path.exists():
        return {}
    with open(preset_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def calculate_num_repeats(image_count: int, num_epochs: int, target_steps: int) -> int:
    """Calculate num_repeats to hit approximately target_steps."""
    if image_count <= 0 or num_epochs <= 0:
        return 10
    repeats = math.ceil(target_steps / (image_count * num_epochs))
    return max(1, min(repeats, 50))  # clamp between 1 and 50


def build_config(
    lora_type: str,
    trigger_word: str,
    base_model: str,
    job_dir: str,
    lora_name: str = "lora_output",
    image_count: int = 0,
    overrides: Optional[Dict[str, Any]] = None,
    gpu_mode: str = "local",
) -> Dict[str, Any]:
    config = load_defaults()
    preset = load_preset(lora_type)
    config = _deep_merge(config, preset)

    if overrides:
        config = _deep_merge(config, overrides)

    # Set model
    config.setdefault("model", {})["base_model"] = base_model

    # Set trigger word
    config.setdefault("caption", {})["trigger_word"] = trigger_word
    config.setdefault("training", {})["trigger_word"] = trigger_word
    config["training"]["lora_name"] = lora_name
    config["training"]["gpu_mode"] = gpu_mode

    # Auto-calculate num_repeats based on image count
    training = config["training"]
    if training.get("auto_num_repeats", True) and image_count > 0:
        target_steps = training.get("target_steps", 2000)
        num_epochs = training.get("num_epochs", 10)
        training["num_repeats"] = calculate_num_repeats(image_count, num_epochs, target_steps)

    # Derive paths
    job = Path(job_dir)
    output_dir = str(job / "output")
    config["data"] = {
        "raw_dir": str(job / "raw"),
        "processed_dir": str(job / "processed"),
        "captions_dir": str(job / "captions"),
        "output_dir": output_dir,
        "lora_path": str(job / "output" / f"{lora_name}.safetensors"),
    }
    config["training"]["output_dir"] = output_dir

    return config

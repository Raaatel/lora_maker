"""Improved training runner with full parameter support for LoRA Maker."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

KOHYA_DIR = Path(os.environ.get("KOHYA_DIR", Path(__file__).resolve().parent.parent.parent / "kohya-sd-scripts"))
TRAIN_SCRIPT = KOHYA_DIR / "sdxl_train_network.py"


def run(
    config: dict[str, Any],
    callback: Callable[[str, dict[str, Any]], None],
    cancel_event: threading.Event,
    pause_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    """Run LoRA training via kohya_ss sd-scripts."""
    try:
        train_cfg = config["training"]
        model_cfg = config["model"]
        data_cfg = config["data"]

        web_app_dir = Path(__file__).resolve().parent.parent.parent
        output_dir = Path(data_cfg["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_dir = Path(data_cfg["processed_dir"]).resolve()
        lora_name = train_cfg.get("lora_name", "lora_output")

        # Kohya dataset dir: {repeats}_{trigger}/
        dataset_dir = output_dir / "dataset"
        num_repeats = train_cfg.get("num_repeats", 10)
        trigger = train_cfg.get("trigger_word", "sks")
        concept_dir = dataset_dir / f"{num_repeats}_{trigger}"
        concept_dir.mkdir(parents=True, exist_ok=True)

        for f in processed_dir.iterdir():
            if f.is_file():
                dest = concept_dir / f.name
                if not dest.exists():
                    shutil.copy2(str(f), str(dest))

        image_count = len([f for f in concept_dir.iterdir()
                           if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
        if image_count == 0:
            callback("error", {"message": "No images found in processed directory"})
            return {"error": "No images"}

        total_steps = image_count * num_repeats * train_cfg.get("num_epochs", 10)
        callback("log", {"message": f"Dataset: {image_count} images × {num_repeats} repeats × {train_cfg.get('num_epochs',10)} epochs = {total_steps} steps", "level": "info"})

        # Python executable
        python_310 = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python310" / "python.exe"
        python_exe = str(python_310) if python_310.exists() else sys.executable

        resolution = train_cfg.get("resolution", 1024)

        # Build base command
        cmd = [
            python_exe, str(TRAIN_SCRIPT),
            "--pretrained_model_name_or_path", model_cfg.get("base_model", ""),
            "--train_data_dir", str(dataset_dir),
            "--output_dir", str(output_dir),
            "--output_name", lora_name,
            "--save_model_as", "safetensors",
            "--save_precision", "fp16",
            "--resolution", f"{resolution},{resolution}",
            "--max_train_epochs", str(train_cfg.get("num_epochs", 10)),
            "--train_batch_size", str(train_cfg.get("train_batch_size", 1)),
            "--network_module", str(train_cfg.get("network_module", "networks.lora")),
            "--network_dim", str(train_cfg.get("lora_rank", 32)),
            "--network_alpha", str(train_cfg.get("lora_alpha", 16)),
        ]
        # LyCORIS / custom network_args
        for arg in train_cfg.get("network_args", []):
            cmd.extend(["--network_args", str(arg)])
        cmd += [
            "--gradient_accumulation_steps", str(train_cfg.get("gradient_accumulation_steps", 1)),
            "--max_grad_norm", str(train_cfg.get("max_grad_norm", 1.0)),
            "--mixed_precision", train_cfg.get("mixed_precision", "fp16"),
            "--seed", str(train_cfg.get("seed", 42)),
            "--caption_extension", ".txt",
            "--save_every_n_epochs", str(train_cfg.get("save_every_n_epochs", 1)),
            "--logging_dir", str(output_dir / "logs"),
        ]

        # Learning rates - separate unet/TE for better control
        unet_lr = train_cfg.get("unet_lr")
        te_lr = train_cfg.get("text_encoder_lr")
        network_train_unet_only = train_cfg.get("network_train_unet_only", False)

        if network_train_unet_only:
            # UNet only mode - single LR
            lr = unet_lr or train_cfg.get("learning_rate", 1e-4)
            cmd.extend(["--learning_rate", str(lr)])
            cmd.append("--network_train_unet_only")
        else:
            # Separate UNet + TE LRs
            if unet_lr:
                cmd.extend(["--unet_lr", str(unet_lr)])
            else:
                cmd.extend(["--learning_rate", str(train_cfg.get("learning_rate", 1e-4))])

            if te_lr:
                if isinstance(te_lr, list):
                    # SDXL has two TEs
                    for lr_val in te_lr:
                        cmd.extend(["--text_encoder_lr", str(lr_val)])
                else:
                    cmd.extend(["--text_encoder_lr", str(te_lr)])

        # LR scheduler
        scheduler = train_cfg.get("lr_scheduler", "cosine_with_restarts")
        cmd.extend(["--lr_scheduler", scheduler])

        warmup = train_cfg.get("lr_warmup_steps", 0)
        if scheduler == "constant":
            warmup = 0
        cmd.extend(["--lr_warmup_steps", str(warmup)])

        if scheduler == "cosine_with_restarts":
            cycles = train_cfg.get("lr_scheduler_num_cycles", 1)
            cmd.extend(["--lr_scheduler_num_cycles", str(cycles)])

        # min_snr_gamma
        if train_cfg.get("min_snr_gamma"):
            cmd.extend(["--min_snr_gamma", str(train_cfg["min_snr_gamma"])])

        # Noise offset
        if train_cfg.get("noise_offset"):
            cmd.extend(["--noise_offset", str(train_cfg["noise_offset"])])

        # Cache latents
        cmd.append("--cache_latents")
        cmd.append("--no_half_vae")

        # Bucket settings
        if train_cfg.get("enable_bucket", True):
            cmd.extend([
                "--enable_bucket",
                "--bucket_reso_steps", str(train_cfg.get("bucket_reso_steps", 64)),
                "--min_bucket_reso", str(train_cfg.get("min_bucket_reso", 256)),
                "--max_bucket_reso", str(train_cfg.get("max_bucket_reso", 2048)),
            ])

        # Caption settings
        if train_cfg.get("shuffle_caption", True):
            cmd.append("--shuffle_caption")
        if train_cfg.get("keep_tokens", 1):
            cmd.extend(["--keep_tokens", str(train_cfg["keep_tokens"])])
        if train_cfg.get("caption_dropout_rate"):
            cmd.extend(["--caption_dropout_rate", str(train_cfg["caption_dropout_rate"])])

        if train_cfg.get("clip_skip", 1) > 1:
            cmd.extend(["--clip_skip", str(train_cfg["clip_skip"])])

        # Performance
        if train_cfg.get("gradient_checkpointing", True):
            cmd.append("--gradient_checkpointing")

        if train_cfg.get("sdpa", True):
            cmd.append("--sdpa")

        optimizer = "AdamW8bit" if train_cfg.get("use_8bit_adam", True) else "AdamW"
        cmd.extend(["--optimizer_type", optimizer])

        # Sample prompt if set
        sample_prompts_path = output_dir / "sample_prompt.txt"
        sample_prompt = train_cfg.get("sample_prompt", "")
        if sample_prompt:
            sample_prompts_path.write_text(sample_prompt, encoding="utf-8")
            cmd.extend([
                "--sample_prompts", str(sample_prompts_path),
                "--sample_sampler", "euler_a",
                "--sample_every_n_epochs", "5",
            ])

        callback("training_start", {"message": "kohya_ss LoRA 학습 시작...", "total_steps": total_steps})
        logger.info("Training command: %s", " ".join(cmd))

        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(KOHYA_DIR),
            env=env,
            encoding="utf-8",
            errors="replace",
        )

        start_time = time.time()
        total_epochs = train_cfg.get("num_epochs", 10)
        last_loss = 0.0
        current_epoch = 0

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            callback("log", {"message": line, "level": "info"})

            if cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                return {"cancelled": True}

            # Parse progress
            step_info = _parse_line(line, callback, total_epochs, start_time)
            if step_info:
                current_epoch = step_info.get("epoch", current_epoch)

            loss_m = re.search(r"(?:avr_)?loss[=:]\s*([\d.]+)", line)
            if loss_m:
                last_loss = float(loss_m.group(1))

            # Detect epoch completion to register checkpoint
            epoch_m = re.search(r"saving checkpoint.*?(\d+)", line, re.IGNORECASE)
            if epoch_m or ("save" in line.lower() and ".safetensors" in line.lower()):
                _scan_and_register_checkpoints(output_dir, lora_name, callback)

        process.wait()

        if process.returncode != 0:
            msg = f"Training failed (exit {process.returncode})"
            callback("error", {"message": msg})
            return {"error": msg}

        # Final checkpoint scan
        _scan_and_register_checkpoints(output_dir, lora_name, callback)

        # Find final output
        final_path = output_dir / f"{lora_name}.safetensors"
        if not final_path.exists():
            for f in sorted(output_dir.glob("*.safetensors")):
                if not re.search(r"-\d+\.safetensors$", f.name):
                    final_path = f
                    break

        # Cleanup dataset staging dir
        shutil.rmtree(str(dataset_dir), ignore_errors=True)

        callback("completed", {"final_loss": last_loss, "lora_path": str(final_path)})
        return {"final_loss": last_loss, "lora_path": str(final_path)}

    except Exception as e:
        logger.exception("Training error")
        callback("error", {"message": str(e)})
        return {"error": str(e)}


def _scan_and_register_checkpoints(output_dir: Path, lora_name: str, callback: Callable) -> None:
    """Scan output dir for epoch checkpoints and notify via callback."""
    pattern = re.compile(rf"{re.escape(lora_name)}-(\d+)\.safetensors$")
    for f in sorted(output_dir.glob("*.safetensors")):
        m = pattern.match(f.name)
        if m:
            epoch = int(m.group(1))
            callback("checkpoint_saved", {
                "epoch": epoch,
                "file_path": str(f),
                "file_name": f.name,
            })


_TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[")
_EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)", re.IGNORECASE)
_LOSS_RE = re.compile(r"(?:avr_)?loss[=:]\s*([\d.]+)")


def _parse_line(line: str, callback: Callable, total_epochs: int, start_time: float) -> Optional[dict]:
    m = _TQDM_RE.search(line)
    if m:
        step = int(m.group(1))
        total = int(m.group(2))
        loss = 0.0
        lm = _LOSS_RE.search(line)
        if lm:
            loss = float(lm.group(1))

        elapsed = time.time() - start_time
        eta = (elapsed / max(step, 1)) * (total - step) if step > 0 else 0

        callback("step", {
            "step": step,
            "total_steps": total,
            "loss": loss,
            "eta_seconds": int(eta),
            "epoch": 0,
            "total_epochs": total_epochs,
        })
        return {"step": step}

    em = _EPOCH_RE.search(line)
    if em:
        epoch = int(em.group(1))
        total = int(em.group(2))
        loss = 0.0
        lm = _LOSS_RE.search(line)
        if lm:
            loss = float(lm.group(1))
        callback("epoch_end", {"epoch": epoch, "total_epochs": total, "avg_loss": loss})
        return {"epoch": epoch}

    return None

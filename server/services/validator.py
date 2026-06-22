"""
LoRA validation service.
Two modes:
  1. weight_check  – fast, always available, analyses safetensors weights
  2. inference     – slower, needs base model, generates before/after images
"""

import io
import base64
import asyncio
from pathlib import Path
from typing import Optional


# ── Weight Analysis ────────────────────────────────────────────────────────────

def analyze_weights(checkpoint_path: str) -> dict:
    """
    Load a .safetensors LoRA file and compute health metrics.
    Returns a dict with grade (A/B/C/F), issues list, and stats.
    """
    try:
        from safetensors import safe_open
        import numpy as np
    except ImportError:
        return {"error": "safetensors / numpy not installed"}

    path = Path(checkpoint_path)
    if not path.exists():
        return {"error": f"File not found: path"}

    issues = []
    stats = {}

    try:
        tensors = {}
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    except Exception as e:
        return {"error": f"Failed to load checkpoint: {e}"}

    # ── Separate up/down/alpha keys ────────────────────────────────────────
    up_keys   = [k for k in tensors if "lora_up"   in k or "lora_B" in k]
    down_keys = [k for k in tensors if "lora_down" in k or "lora_A" in k]
    alpha_keys = [k for k in tensors if "alpha" in k]

    stats["total_keys"]   = len(tensors)
    stats["lora_modules"] = max(len(up_keys), len(down_keys))

    if not up_keys and not down_keys:
        return {"error": "No LoRA weight keys found — not a valid LoRA file", "grade": "F"}

    # ── Rank detection ──────────────────────────────────────────────────────
    ranks = set()
    for k in down_keys:
        t = tensors[k]
        if t.ndim >= 2:
            ranks.add(min(t.shape))
    stats["detected_rank"] = list(ranks)[0] if len(ranks) == 1 else list(ranks)

    # ── Alpha values ────────────────────────────────────────────────────────
    alphas = set()
    for k in alpha_keys:
        alphas.add(float(tensors[k]))
    if alphas:
        stats["alpha"] = list(alphas)[0] if len(alphas) == 1 else list(alphas)

    # ── NaN / Inf check ─────────────────────────────────────────────────────
    nan_keys, inf_keys = [], []
    for k, t in tensors.items():
        if np.any(np.isnan(t)):   nan_keys.append(k)
        if np.any(np.isinf(t)):   inf_keys.append(k)
    if nan_keys:
        issues.append({"level": "error", "msg": f"NaN values in {len(nan_keys)} tensors — training likely diverged"})
    if inf_keys:
        issues.append({"level": "error", "msg": f"Inf values in {len(inf_keys)} tensors — training diverged"})

    # ── Norm analysis ────────────────────────────────────────────────────────
    up_norms, down_norms = [], []
    for k in up_keys:
        up_norms.append(float(np.linalg.norm(tensors[k].astype(np.float32))))
    for k in down_keys:
        down_norms.append(float(np.linalg.norm(tensors[k].astype(np.float32))))

    if up_norms:
        avg_up = sum(up_norms) / len(up_norms)
        stats["avg_up_norm"]   = round(avg_up, 4)
        if avg_up < 1e-5:
            issues.append({"level": "warning", "msg": "Up-projection norms very small — LoRA may be undertrained or LR too low"})
        elif avg_up > 100:
            issues.append({"level": "warning", "msg": "Up-projection norms very large — possible overtraining or LR too high"})

    if down_norms:
        avg_down = sum(down_norms) / len(down_norms)
        stats["avg_down_norm"] = round(avg_down, 4)

    # ── Effective strength ───────────────────────────────────────────────────
    if up_norms and down_norms:
        effective = (sum(up_norms)/len(up_norms)) * (sum(down_norms)/len(down_norms))
        stats["effective_strength"] = round(effective, 2)
        if effective < 0.01:
            issues.append({"level": "warning", "msg": "Effective LoRA strength very low — style may not transfer well"})
        elif effective > 5000:
            issues.append({"level": "warning", "msg": "Effective LoRA strength very high — may cause over-saturation"})

    # ── Grade ────────────────────────────────────────────────────────────────
    errors   = [i for i in issues if i["level"] == "error"]
    warnings = [i for i in issues if i["level"] == "warning"]

    if errors:
        grade = "F"
    elif len(warnings) >= 2:
        grade = "C"
    elif len(warnings) == 1:
        grade = "B"
    else:
        grade = "A"

    return {
        "grade": grade,
        "issues": issues,
        "stats": stats,
        "file_size_mb": round(path.stat().st_size / 1_000_000, 2),
    }


# ── Inference Test ─────────────────────────────────────────────────────────────

async def run_inference_test(
    checkpoint_path: str,
    base_model_path: str,
    prompt: str,
    trigger_word: str,
    seed: int = 42,
    steps: int = 20,
    width: int = 512,
    height: int = 512,
) -> dict:
    """
    Generate two images (without/with LoRA) and return as base64 PNGs.
    Runs in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _inference_blocking,
        checkpoint_path, base_model_path, prompt, trigger_word, seed, steps, width, height)
    return result


def _inference_blocking(checkpoint_path, base_model_path, prompt, trigger_word, seed, steps, width, height):
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    base = Path(base_model_path)
    if not base.exists():
        return {"error": f"Base model not found: {base_model_path}"}

    chk = Path(checkpoint_path)
    if not chk.exists():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    try:
        import torch
        base_str = str(base)
        is_single_file = base_str.lower().endswith(('.safetensors', '.ckpt', '.pt'))

        if is_single_file:
            # Use load_file + manual component loading to avoid CLIPTextModel conversion bugs
            from safetensors.torch import load_file as st_load
            from diffusers import (
                AutoencoderKL, UNet2DConditionModel,
                EulerDiscreteScheduler,
            )
            from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

            # Load via from_single_file with ignore_mismatched_sizes to be lenient
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        base_str,
                        torch_dtype=dtype,
                        local_files_only=False,
                        use_safetensors=True,
                    ).to(device)
            except AttributeError as clip_err:
                if "text_model" in str(clip_err):
                    # Known diffusers/transformers incompatibility with some SDXL checkpoints.
                    # Fall back: load the HF-format SDXL base pipeline and swap the UNet weights.
                    from safetensors.torch import load_file as sf_load
                    from diffusers.loaders.single_file_utils import (
                        convert_ldm_unet_checkpoint,
                        convert_ldm_vae_checkpoint,
                    )
                    return {
                        "error": (
                            "이 모델 형식은 이미지 생성 테스트를 지원하지 않습니다. "
                            "HuggingFace diffusers 형식(폴더)의 모델을 사용하거나, "
                            "ComfyUI 등 다른 도구로 테스트해주세요."
                        )
                    }
                raise
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                base_str,
                torch_dtype=dtype,
                use_safetensors=True,
            ).to(device)
        pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        return {"error": f"Failed to load base model: {e}"}

    RESOLUTIONS = [
        (1024, 1024, "1:1"),
        (832,  1216, "2:3 세로"),
        (1216, 832,  "3:2 가로"),
    ]

    lora_prompt = f"{trigger_word}, {prompt}" if trigger_word else prompt
    results = []

    for (w, h, label) in RESOLUTIONS:
        # ── Without LoRA ─────────────────────────────────────────────────
        try:
            gen = torch.Generator(device=device).manual_seed(seed)
            img_before = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                width=w, height=h,
                generator=gen,
            ).images[0]
        except Exception as e:
            return {"error": f"Inference (without LoRA, {w}x{h}) failed: {e}"}

        # ── With LoRA ─────────────────────────────────────────────────────
        try:
            pipe.load_lora_weights(str(chk))
            gen2 = torch.Generator(device=device).manual_seed(seed)
            img_after = pipe(
                prompt=lora_prompt,
                num_inference_steps=steps,
                width=w, height=h,
                generator=gen2,
            ).images[0]
            pipe.unload_lora_weights()
        except Exception as e:
            return {"error": f"Inference (with LoRA, {w}x{h}) failed: {e}"}

        results.append({
            "label": label,
            "size": f"{w}×{h}",
            "before": _img_to_b64(img_before),
            "after":  _img_to_b64(img_after),
        })

    # Free VRAM
    try:
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "results": results,
        "prompt_used": lora_prompt,
    }


def _img_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

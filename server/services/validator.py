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

# ── Generation progress/cancel state ──────────────────────────────────────────
_progress_store: dict = {}  # gen_id -> {"percent": int, "label": str}
_cancel_store:   dict = {}  # gen_id -> bool

# ── Model cache (avoids reload on every generation) ───────────────────────────
import threading as _threading
_pipe_cache:      dict = {}   # base_model_path -> pipe
_pipe_cache_lock: object = _threading.Lock()


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
    cfg_scale: float = 7.0,
    scheduler: str = "euler",
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    resolutions: list = None,
    gen_id: str = "",
    lora_scale: float = 1.0,
    denoising_strength: float = 0.75,
    input_image_path: str = "",
) -> dict:
    """
    Generate two images (without/with LoRA) and return as base64 PNGs.
    Runs in a thread pool to avoid blocking the event loop.
    """
    import functools
    if gen_id:
        _progress_store[gen_id] = {"percent": 0, "label": "준비 중..."}
        _cancel_store.pop(gen_id, None)
    loop = asyncio.get_event_loop()
    fn = functools.partial(_inference_blocking,
        checkpoint_path, base_model_path, prompt, trigger_word,
        seed, steps, cfg_scale, scheduler, negative_prompt, width, height,
        resolutions, gen_id, lora_scale, denoising_strength, input_image_path)
    result = await loop.run_in_executor(None, fn)
    if gen_id:
        _progress_store[gen_id] = {"percent": 100, "label": "완료!"}
    return result


def _inference_blocking(checkpoint_path, base_model_path, prompt, trigger_word,
                        seed, steps, cfg_scale, scheduler, negative_prompt, width, height,
                        resolutions=None, gen_id="", lora_scale=1.0,
                        denoising_strength=0.75, input_image_path=""):
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

    from diffusers import (
        StableDiffusionXLPipeline,
        EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler, DDIMScheduler,
        HeunDiscreteScheduler, LMSDiscreteScheduler,
    )
    _sched_map = {
        "euler":           EulerDiscreteScheduler,
        "euler_a":         EulerAncestralDiscreteScheduler,
        "dpm++_2m":        DPMSolverMultistepScheduler,
        "dpm++_2m_karras": DPMSolverMultistepScheduler,
        "ddim":            DDIMScheduler,
        "heun":            HeunDiscreteScheduler,
        "lms":             LMSDiscreteScheduler,
    }

    # ── Model cache check ────────────────────────────────────────────────
    cache_key = str(base)
    pipe = None
    with _pipe_cache_lock:
        if cache_key in _pipe_cache:
            pipe = _pipe_cache[cache_key]

    if pipe is None:
        if gen_id:
            _progress_store[gen_id] = {"percent": 2, "label": "모델 로딩 중... (최초 1회, 이후 캐시됨)"}
        try:
            base_str = str(base)
            is_single_file = base_str.lower().endswith(('.safetensors', '.ckpt', '.pt'))
            if is_single_file:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pipe = StableDiffusionXLPipeline.from_single_file(
                            base_str, torch_dtype=dtype,
                            local_files_only=False, use_safetensors=True,
                        ).to(device)
                except AttributeError as clip_err:
                    if "text_model" in str(clip_err):
                        return {"error": "이 모델 형식은 이미지 생성 테스트를 지원하지 않습니다. HuggingFace diffusers 형식(폴더)을 사용해주세요."}
                    raise
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    base_str, torch_dtype=dtype, use_safetensors=True,
                ).to(device)

            # Speed optimizations
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                try:
                    pipe.enable_attention_slicing(1)
                except Exception:
                    pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
            pipe.set_progress_bar_config(disable=True)

            with _pipe_cache_lock:
                _pipe_cache[cache_key] = pipe
        except Exception as e:
            return {"error": f"Failed to load base model: {e}"}
    else:
        if gen_id:
            _progress_store[gen_id] = {"percent": 3, "label": "모델 캐시 적중 · 바로 생성 시작!"}

    # ── Apply scheduler (per-request, not cached) ──────────────────────
    try:
        sched_cls = _sched_map.get(scheduler, EulerDiscreteScheduler)
        karras = scheduler == "dpm++_2m_karras"
        if karras:
            pipe.scheduler = sched_cls.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        else:
            pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
    except Exception:
        pass

    if gen_id:
        _progress_store[gen_id] = {"percent": 5, "label": "생성 시작..."}

    if resolutions:
        RESOLUTIONS = [(r['width'], r['height'], r.get('label', f"{r['width']}x{r['height']}")) for r in resolutions]
    else:
        RESOLUTIONS = [
            (1024, 1024, "1:1"),
            (832,  1216, "2:3 세로"),
            (1216, 832,  "3:2 가로"),
        ]

    total_passes = len(RESOLUTIONS) * 2  # before + after per resolution

    def _make_cb(pass_idx, phase_label, size_label):
        """Returns a diffusers step callback that updates progress and checks cancel."""
        def cb(pipe, step, timestep, kwargs):
            if gen_id and _cancel_store.get(gen_id):
                raise InterruptedError("사용자가 취소했습니다")
            if gen_id:
                base = 5 + pass_idx * (95 / total_passes)
                pct  = base + (step / max(steps, 1)) * (95 / total_passes)
                _progress_store[gen_id] = {
                    "percent": int(min(pct, 99)),
                    "label": f"{phase_label} · {size_label} ({step}/{steps}스텝)"
                }
            return kwargs
        return cb

    if gen_id:
        _progress_store[gen_id] = {"percent": 3, "label": "모델 로딩 중..."}

    lora_prompt = f"{trigger_word}, {prompt}" if trigger_word else prompt
    # before/after 모두 동일한 프롬프트 사용 (trigger word 포함)
    # → 프롬프트 조건을 동일하게 유지, LoRA 유무만 차이
    before_prompt = lora_prompt

    # ── seed 확정: before/after 동일 seed 사용해야 LoRA 효과만 비교 가능 ──
    effective_seed = seed if seed >= 0 else torch.randint(0, 2**31 - 1, (1,)).item()

    results = []
    before_images = []  # (w, h, label, img) 저장

    # ── Phase 1: 모든 해상도 before 생성 (LoRA 없는 깨끗한 상태) ──────────
    for res_idx, (w, h, label) in enumerate(RESOLUTIONS):
        if gen_id and _cancel_store.get(gen_id):
            return {"error": "사용자가 취소했습니다", "cancelled": True}
        size_label = f"{w}×{h}"
        pass_before = res_idx * 2
        if gen_id:
            _progress_store[gen_id] = {
                "percent": int(5 + pass_before * (95 / total_passes)),
                "label": f"베이스 이미지 생성 중 · {size_label} (seed {effective_seed})"
            }
        try:
            gen = torch.Generator(device=device).manual_seed(effective_seed)
            _cb = _make_cb(pass_before, "베이스", size_label)
            _kwargs = dict(
                prompt=before_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=w, height=h,
                generator=gen,
            )
            if negative_prompt:
                _kwargs["negative_prompt"] = negative_prompt
            try:
                img_before = pipe(**_kwargs,
                    callback_on_step_end=_cb,
                    callback_on_step_end_tensor_inputs=["latents"]).images[0]
            except TypeError:
                img_before = pipe(**_kwargs).images[0]
            before_images.append((w, h, label, img_before))
        except InterruptedError:
            return {"error": "사용자가 취소했습니다", "cancelled": True}
        except Exception as e:
            return {"error": f"Inference (without LoRA, {w}x{h}) failed: {e}"}

    # ── Phase 2: LoRA 한 번만 로드 후 전체 after 생성 ─────────────────────
    try:
        pipe.load_lora_weights(str(chk))
    except Exception as e:
        return {"error": f"LoRA 로드 실패: {e}"}

    try:
        for res_idx, (w, h, label, img_before) in enumerate(before_images):
            if gen_id and _cancel_store.get(gen_id):
                return {"error": "사용자가 취소했습니다", "cancelled": True}
            size_label = f"{w}×{h}"
            pass_after = res_idx * 2 + 1
            if gen_id:
                _progress_store[gen_id] = {
                    "percent": int(5 + pass_after * (95 / total_passes)),
                    "label": f"LoRA 적용 이미지 생성 중 · {size_label} (seed {effective_seed})"
                }
            try:
                gen2 = torch.Generator(device=device).manual_seed(effective_seed)
                _cb2 = _make_cb(pass_after, "LoRA 적용", size_label)
                _kwargs2 = dict(
                    prompt=lora_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=w, height=h,
                    generator=gen2,
                    cross_attention_kwargs={"scale": lora_scale},
                )
                if negative_prompt:
                    _kwargs2["negative_prompt"] = negative_prompt
                # img2img mode
                if input_image_path:
                    from diffusers import StableDiffusionXLImg2ImgPipeline
                    from PIL import Image as _PILImage
                    try:
                        _init_img = _PILImage.open(input_image_path).convert("RGB").resize((w, h))
                        _img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
                        _i2i_kwargs = dict(
                            prompt=lora_prompt,
                            image=_init_img,
                            strength=denoising_strength,
                            num_inference_steps=steps,
                            guidance_scale=cfg_scale,
                            generator=gen2,
                            cross_attention_kwargs={"scale": lora_scale},
                        )
                        if negative_prompt:
                            _i2i_kwargs["negative_prompt"] = negative_prompt
                        img_after = _img2img_pipe(**_i2i_kwargs).images[0]
                    except Exception as e2i_err:
                        return {"error": f"img2img failed: {e2i_err}"}
                else:
                    try:
                        img_after = pipe(**_kwargs2,
                            callback_on_step_end=_cb2,
                            callback_on_step_end_tensor_inputs=["latents"]).images[0]
                    except TypeError:
                        img_after = pipe(**_kwargs2).images[0]
            except InterruptedError:
                return {"error": "사용자가 취소했습니다", "cancelled": True}
            except Exception as e:
                return {"error": f"Inference (with LoRA, {w}x{h}) failed: {e}"}

            results.append({
                "label": label,
                "size": f"{w}×{h}",
                "before": _img_to_b64(img_before),
                "after":  _img_to_b64(img_after),
            })
    finally:
        # ── LoRA 반드시 언로드 (캐시된 파이프 오염 방지) ──────────────────
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
    # VRAM 캐시 정리 (모델은 캐시에 유지)
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "results": results,
        "prompt_used": lora_prompt,
        "seed_used": effective_seed,
    }


def _img_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

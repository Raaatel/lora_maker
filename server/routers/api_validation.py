"""Validation API — weight analysis and optional inference test."""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from server import database as db
from server.services.validator import analyze_weights, run_inference_test

router = APIRouter(tags=["validation"])

# ── Standalone (no project) endpoints ─────────────────────────────────────────

class FileValidateRequest(BaseModel):
    file_path: str

class FileInferenceRequest(BaseModel):
    file_path: str
    base_model: str
    prompt: str = "masterpiece, best quality, 1girl, portrait"
    negative_prompt: str = ""
    trigger_word: str = ""
    seed: int = 42
    steps: int = 20
    cfg_scale: float = 7.0
    scheduler: str = "euler"
    width: int = 512
    height: int = 512
    resolutions: Optional[list] = None
    gen_id: str = ""
    lora_scale: float = 1.0
    denoising_strength: float = 0.75
    input_image_path: str = "" 

@router.post("/api/validate/file")
async def validate_file(body: FileValidateRequest):
    """Analyze weights of any .safetensors file by path."""
    result = analyze_weights(body.file_path)
    return JSONResponse(result)

@router.post("/api/validate/inference-file")
async def validate_inference_file(body: FileInferenceRequest):
    """Generate before/after images for any .safetensors file."""
    result = await run_inference_test(
        checkpoint_path=body.file_path,
        base_model_path=body.base_model,
        prompt=body.prompt,
        trigger_word=body.trigger_word,
        seed=body.seed,
        steps=body.steps,
        cfg_scale=body.cfg_scale,
        scheduler=body.scheduler,
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        resolutions=body.resolutions,
        gen_id=body.gen_id,
        lora_scale=body.lora_scale,
        denoising_strength=body.denoising_strength,
        input_image_path=body.input_image_path,
    )
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(result)

# ── Per-project checkpoint endpoints ──────────────────────────────────────────

_project_router = APIRouter(prefix="/api/projects/{project_id}/validate", tags=["validation"])
DATA_DIR = Path("data/jobs")

# Cache validation results in memory (project_id -> epoch -> result)
_cache: dict = {}


@_project_router.get("/weight/{epoch}")
async def validate_weight(project_id: str, epoch: str):
    """Fast weight analysis for a specific checkpoint epoch."""
    # Guard: epoch must be numeric
    if not epoch.lstrip('-').isdigit():
        raise HTTPException(status_code=422, detail=f"epoch must be an integer, got '{epoch}'")
    epoch = int(epoch)  # type: ignore
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    cache_key = (project_id, epoch, "weight")
    if cache_key in _cache:
        return JSONResponse(_cache[cache_key])

    chk = await db.get_checkpoint(project_id, epoch)
    if not chk:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    file_path = chk.get("file_path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Checkpoint file missing")

    result = analyze_weights(file_path)
    _cache[cache_key] = result
    return JSONResponse(result)


class InferenceRequest(BaseModel):
    epoch: int
    base_model: str
    prompt: str = "masterpiece, best quality, 1girl, portrait"
    negative_prompt: str = ""
    seed: int = 42
    steps: int = 20
    cfg_scale: float = 7.0
    scheduler: str = "euler"
    width: int = 512
    height: int = 512


@_project_router.post("/inference")
async def validate_inference(project_id: str, body: InferenceRequest):
    """
    Run a before/after inference test with the LoRA checkpoint.
    Returns base64 PNG images. Can take 30-120s depending on hardware.
    """
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    chk = await db.get_checkpoint(project_id, body.epoch)
    if not chk:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    file_path = chk.get("file_path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Checkpoint file missing")

    result = await run_inference_test(
        checkpoint_path=file_path,
        base_model_path=body.base_model,
        prompt=body.prompt,
        trigger_word=project.get("trigger_word", ""),
        seed=body.seed,
        steps=body.steps,
        cfg_scale=body.cfg_scale,
        scheduler=body.scheduler,
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return JSONResponse(result)

# Register project-scoped routes onto the main router
router.include_router(_project_router)


# ── Generation progress / cancel ───────────────────────────────────────────────

@router.get("/api/generate/progress/{gen_id}")
async def get_gen_progress(gen_id: str):
    from server.services.validator import _progress_store
    return JSONResponse(_progress_store.get(gen_id, {"percent": 0, "label": "대기 중..."}))


@router.post("/api/generate/cancel/{gen_id}")
async def cancel_gen(gen_id: str):
    from server.services.validator import _cancel_store
    _cancel_store[gen_id] = True
    return JSONResponse({"ok": True})

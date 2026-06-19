"""Project management API."""

import asyncio
import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from server import database as db
from server.services.config_builder import build_config, load_preset

router = APIRouter(prefix="/api/projects", tags=["projects"])

DATA_DIR = Path("data/jobs")


class ProjectCreate(BaseModel):
    name: str
    lora_type: str
    trigger_word: str
    base_model: str
    gpu_mode: str = "local"
    config_overrides: dict = {}


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    base_model: Optional[str] = None


@router.post("")
async def create_project(body: ProjectCreate):
    project_id = str(uuid.uuid4())
    job_dir = DATA_DIR / project_id

    for subdir in ("raw", "processed", "captions", "output", "thumbnails"):
        (job_dir / subdir).mkdir(parents=True, exist_ok=True)

    config = build_config(
        lora_type=body.lora_type,
        trigger_word=body.trigger_word,
        base_model=body.base_model,
        job_dir=str(job_dir),
        lora_name=body.name.replace(" ", "_"),
        gpu_mode=body.gpu_mode,
        overrides=body.config_overrides or {},
    )

    project = await db.create_project(
        project_id=project_id,
        name=body.name,
        lora_type=body.lora_type,
        trigger_word=body.trigger_word,
        base_model=body.base_model,
        config_json=json.dumps(config, ensure_ascii=False),
        gpu_mode=body.gpu_mode,
    )

    return JSONResponse(project)


@router.get("")
async def list_projects():
    projects = await db.list_projects()
    return JSONResponse(projects)


@router.get("/{project_id}")
async def get_project(project_id: str):
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return JSONResponse(project)


@router.post("/{project_id}/run")
async def run_project(project_id: str, request: Request):
    """Start the training pipeline."""
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    job_dir = DATA_DIR / project_id
    raw_dir = job_dir / "raw"
    image_count = len([
        f for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]) if raw_dir.exists() else 0

    if image_count == 0:
        raise HTTPException(status_code=400, detail="이미지가 없습니다. 먼저 이미지를 업로드하세요.")

    # Rebuild config with correct image count for auto num_repeats
    config_data = json.loads(project["config_json"])
    fresh_config = build_config(
        lora_type=project["lora_type"],
        trigger_word=project["trigger_word"],
        base_model=project["base_model"],
        job_dir=str(job_dir),
        lora_name=project["name"].replace(" ", "_"),
        image_count=image_count,
        gpu_mode=project["gpu_mode"],
    )

    # Apply any saved overrides
    import copy
    saved_training = config_data.get("training", {})
    # Keep user-customized settings but recalculate num_repeats
    for k, v in saved_training.items():
        if k not in ("num_repeats", "output_dir", "lora_name", "trigger_word"):
            fresh_config["training"][k] = v

    await db.update_project_status(project_id, "pending", image_count=image_count)

    training_manager = request.app.state.training_manager
    asyncio.create_task(training_manager.start_training(project_id, fresh_config))

    return JSONResponse({
        "status": "starting",
        "image_count": image_count,
        "num_repeats": fresh_config["training"].get("num_repeats"),
        "total_steps": image_count * fresh_config["training"].get("num_repeats", 1) * fresh_config["training"].get("num_epochs", 10),
    })


@router.post("/{project_id}/pause")
async def pause_project(project_id: str, request: Request):
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    ok = request.app.state.training_manager.pause_training()
    if not ok:
        raise HTTPException(status_code=400, detail="학습 중이 아닙니다")
    return JSONResponse({"status": "paused"})


@router.post("/{project_id}/resume")
async def resume_project(project_id: str, request: Request):
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    ok = request.app.state.training_manager.resume_training()
    if not ok:
        raise HTTPException(status_code=400, detail="일시정지 상태가 아닙니다")
    return JSONResponse({"status": "resumed"})


@router.post("/{project_id}/cancel")
async def cancel_project(project_id: str, request: Request):
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await request.app.state.training_manager.cancel_training()
    return JSONResponse({"status": "cancelled"})


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await db.delete_project(project_id)
    job_dir = DATA_DIR / project_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    return JSONResponse({"status": "deleted"})


@router.get("/{project_id}/config")
async def get_config(project_id: str):
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    config = json.loads(project["config_json"]) if project.get("config_json") else {}
    return JSONResponse(config)


@router.get("/presets/{lora_type}")
async def get_preset(lora_type: str):
    preset = load_preset(lora_type)
    return JSONResponse(preset)

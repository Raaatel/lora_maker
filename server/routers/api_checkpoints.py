"""Checkpoint management API."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from server import database as db
from server.services.checkpoint_manager import sync_checkpoints

router = APIRouter(prefix="/api/projects/{project_id}/checkpoints", tags=["checkpoints"])

DATA_DIR = Path("data/jobs")


@router.get("")
async def list_checkpoints(project_id: str):
    """List all checkpoints for a project, syncing from disk first."""
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    job_dir = DATA_DIR / project_id
    output_dir = str(job_dir / "output")
    lora_name = project["name"].replace(" ", "_")

    checkpoints = await sync_checkpoints(project_id, output_dir, lora_name)
    return JSONResponse(checkpoints)


# ── Static keyword routes MUST come before /{epoch} to avoid int-parse errors ─

@router.get("/best/download")
async def download_best_checkpoint(project_id: str):
    """Download the checkpoint with the best (lowest) loss."""
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    checkpoints = await db.list_checkpoints(project_id)
    if not checkpoints:
        raise HTTPException(status_code=404, detail="No checkpoints found")

    best = next((c for c in checkpoints if c.get("is_best")), None)
    if not best:
        with_loss = [c for c in checkpoints if c.get("loss") is not None]
        if with_loss:
            best = min(with_loss, key=lambda c: c["loss"])
        else:
            best = checkpoints[-1]

    file_path = Path(best["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Best checkpoint file not found")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@router.get("/final/download")
async def download_final(project_id: str):
    """Download the final merged LoRA file."""
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    lora_path = project.get("lora_file_path")
    if not lora_path or not Path(lora_path).exists():
        job_dir = DATA_DIR / project_id
        lora_name = project["name"].replace(" ", "_")
        lora_path = str(job_dir / "output" / f"{lora_name}.safetensors")
        if not Path(lora_path).exists():
            raise HTTPException(status_code=404, detail="최종 LoRA 파일이 없습니다")

    file_path = Path(lora_path)
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


# ── Parameterised epoch route ─────────────────────────────────────────────────

@router.get("/{epoch}/download")
async def download_checkpoint(project_id: str, epoch: int):
    """Download a specific epoch checkpoint."""
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    chk = await db.get_checkpoint(project_id, epoch)
    if not chk:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    file_path = Path(chk["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint file not found on disk")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )

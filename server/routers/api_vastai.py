"""Vast.ai integration API."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from server import database as db
from server.services.vastai_service import search_instances, VastAIError

router = APIRouter(prefix="/api/vastai", tags=["vastai"])


class VastAISettings(BaseModel):
    api_key: str
    ssh_key_path: Optional[str] = ""


@router.post("/settings")
async def save_settings(body: VastAISettings):
    """Save Vast.ai API key and SSH key path."""
    await db.set_setting("vastai_api_key", body.api_key)
    if body.ssh_key_path:
        await db.set_setting("vastai_ssh_key_path", body.ssh_key_path)
    return JSONResponse({"status": "saved"})


@router.get("/settings")
async def get_settings():
    """Get current Vast.ai settings (key is masked)."""
    api_key = await db.get_setting("vastai_api_key") or ""
    ssh_key = await db.get_setting("vastai_ssh_key_path") or ""
    return JSONResponse({
        "has_api_key": bool(api_key),
        "api_key_preview": f"...{api_key[-6:]}" if len(api_key) > 6 else "",
        "ssh_key_path": ssh_key,
    })


@router.get("/instances")
async def list_instances(min_vram: float = 20, gpu_name: Optional[str] = None):
    """Search for available GPU instances."""
    api_key = await db.get_setting("vastai_api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Vast.ai API 키가 설정되지 않았습니다")

    try:
        instances = await search_instances(api_key, min_vram_gb=min_vram, gpu_name=gpu_name)
        # Return simplified list
        return JSONResponse([{
            "id": inst.get("id"),
            "gpu_name": inst.get("gpu_name"),
            "num_gpus": inst.get("num_gpus", 1),
            "gpu_ram_gb": round(inst.get("gpu_ram", 0) / 1024, 1),
            "price_per_hour": round(inst.get("dph_total", 0), 4),
            "reliability": round(inst.get("reliability2", 0) * 100, 1),
            "location": inst.get("geolocation", "Unknown"),
            "disk_space_gb": inst.get("disk_space", 0),
        } for inst in instances[:20]])
    except VastAIError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status")
async def connection_status():
    """Check if Vast.ai API key is valid."""
    api_key = await db.get_setting("vastai_api_key")
    if not api_key:
        return JSONResponse({"connected": False, "message": "API 키 없음"})
    try:
        await search_instances(api_key, min_vram_gb=16)
        return JSONResponse({"connected": True})
    except VastAIError as e:
        return JSONResponse({"connected": False, "message": str(e)})

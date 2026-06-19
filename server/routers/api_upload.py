"""Image upload API."""

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

router = APIRouter(prefix="/api/projects/{project_id}/images", tags=["upload"])

DATA_DIR = Path("data/jobs")
ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
THUMB_SIZE = 256


@router.post("")
async def upload_images(project_id: str, files: List[UploadFile] = File(...)):
    raw_dir = DATA_DIR / project_id / "raw"
    thumb_dir = DATA_DIR / project_id / "thumbnails"

    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    thumb_dir.mkdir(exist_ok=True)

    results = []
    for upload in files:
        ext = Path(upload.filename).suffix.lower()
        if ext not in ALLOWED:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 형식: {ext}")

        content = await upload.read()
        file_path = raw_dir / upload.filename
        file_path.write_bytes(content)

        try:
            with Image.open(file_path) as img:
                img.thumbnail((THUMB_SIZE, THUMB_SIZE))
                thumb = thumb_dir / upload.filename
                img.save(thumb, format="PNG")
        except Exception:
            pass

        results.append({
            "filename": upload.filename,
            "thumbnail_url": f"/files/{project_id}/thumbnails/{upload.filename}",
            "size": len(content),
        })

    return JSONResponse(results)


@router.get("")
async def list_images(project_id: str):
    raw_dir = DATA_DIR / project_id / "raw"
    thumb_dir = DATA_DIR / project_id / "thumbnails"

    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    results = []
    for f in sorted(raw_dir.iterdir()):
        if f.suffix.lower() in ALLOWED:
            thumb = thumb_dir / f.name
            results.append({
                "filename": f.name,
                "thumbnail_url": f"/files/{project_id}/thumbnails/{f.name}" if thumb.exists() else None,
                "size": f.stat().st_size,
            })
    return JSONResponse(results)


@router.delete("/{filename}")
async def delete_image(project_id: str, filename: str):
    raw_dir = DATA_DIR / project_id / "raw"
    thumb_dir = DATA_DIR / project_id / "thumbnails"

    file_path = raw_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    file_path.unlink()
    thumb = thumb_dir / filename
    if thumb.exists():
        thumb.unlink()

    return JSONResponse({"status": "deleted"})

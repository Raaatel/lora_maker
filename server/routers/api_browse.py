"""Native file/folder browse dialog via tkinter (local server only)."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter(tags=["browse"])


class BrowseRequest(BaseModel):
    mode: str = "file"          # "file" | "folder"
    accept: str = ""            # e.g. ".safetensors,.ckpt"
    title: str = "선택"


@router.post("/api/browse")
async def browse_path(body: BrowseRequest):
    """
    Open a native OS file/folder picker via tkinter.
    Returns { path: str } or { path: null } if cancelled.
    Only works when the server runs locally (not in a container).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        if body.mode == "folder":
            path = filedialog.askdirectory(title=body.title, parent=root)
        else:
            # Build filetypes from accept string
            filetypes = []
            if body.accept:
                exts = [e.strip().lstrip('.') for e in body.accept.split(',') if e.strip()]
                ext_pattern = ' '.join(f'*.{e}' for e in exts)
                filetypes.append((f"{', '.join(exts).upper()} 파일", ext_pattern))
            filetypes.append(("모든 파일", "*.*"))

            path = filedialog.askopenfilename(
                title=body.title,
                filetypes=filetypes,
                parent=root,
            )

        root.destroy()
        return JSONResponse({"path": path if path else None})

    except Exception as e:
        return JSONResponse({"path": None, "error": str(e)})


# ── Local image serving ──────────────────────────────────────────────────────
import os, mimetypes
from fastapi.responses import FileResponse, Response

@router.get("/api/image")
async def serve_local_image(path: str):
    """Serve a local image file so the browser can display it (avoids file:// CORS)."""
    if not path or not os.path.isfile(path):
        return Response(status_code=404)
    mime, _ = mimetypes.guess_type(path)
    return FileResponse(path, media_type=mime or "image/png")

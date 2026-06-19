"""Checkpoint manager - scans output dirs and syncs to DB."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from server import database as db


async def sync_checkpoints(project_id: str, output_dir: str, lora_name: str) -> list[dict[str, Any]]:
    """Scan output_dir for epoch checkpoint files and sync to DB."""
    out = Path(output_dir)
    if not out.exists():
        return []

    pattern = re.compile(rf"{re.escape(lora_name)}-(\d+)\.safetensors$")
    found = []

    for f in sorted(out.glob("*.safetensors")):
        m = pattern.match(f.name)
        if m:
            epoch = int(m.group(1))
            found.append((epoch, f))

    if not found:
        return await db.list_checkpoints(project_id)

    # Get existing checkpoints to compare losses
    existing = {c["epoch"]: c for c in await db.list_checkpoints(project_id)}
    result = []

    for epoch, path in found:
        existing_chk = existing.get(epoch)
        loss = existing_chk["loss"] if existing_chk else None
        chk = await db.upsert_checkpoint(
            project_id=project_id,
            epoch=epoch,
            file_path=str(path),
            loss=loss,
            is_best=False,
        )
        result.append(chk)

    # Mark best loss checkpoint
    checkpoints_with_loss = [c for c in result if c.get("loss") is not None]
    if checkpoints_with_loss:
        best = min(checkpoints_with_loss, key=lambda c: c["loss"])
        await db.upsert_checkpoint(
            project_id=project_id,
            epoch=best["epoch"],
            file_path=best["file_path"],
            loss=best["loss"],
            is_best=True,
        )

    return await db.list_checkpoints(project_id)


async def register_checkpoint_from_event(
    project_id: str,
    epoch: int,
    file_path: str,
    loss: Optional[float] = None,
) -> dict[str, Any]:
    """Register a checkpoint immediately when saved during training."""
    return await db.upsert_checkpoint(
        project_id=project_id,
        epoch=epoch,
        file_path=file_path,
        loss=loss,
        is_best=False,
    )


def get_checkpoint_file(output_dir: str, lora_name: str, epoch: int) -> Optional[Path]:
    out = Path(output_dir)
    path = out / f"{lora_name}-{epoch:06d}.safetensors"
    if path.exists():
        return path
    # Try other zero-padding variants
    for f in out.glob(f"{lora_name}-*.safetensors"):
        m = re.search(r"-(\d+)\.safetensors$", f.name)
        if m and int(m.group(1)) == epoch:
            return f
    return None

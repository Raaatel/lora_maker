"""SQLite database layer for LoRA Maker."""

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import aiosqlite

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "db"
DB_PATH = DB_DIR / "app.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    lora_type       TEXT NOT NULL,
    trigger_word    TEXT NOT NULL,
    base_model      TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',
    config_json     TEXT,
    image_count     INTEGER DEFAULT 0,
    current_epoch   INTEGER DEFAULT 0,
    total_epochs    INTEGER DEFAULT 0,
    current_step    INTEGER DEFAULT 0,
    total_steps     INTEGER DEFAULT 0,
    current_loss    REAL,
    best_loss       REAL,
    best_epoch      INTEGER,
    eta_seconds     INTEGER,
    error_message   TEXT,
    lora_file_path  TEXT,
    output_dir      TEXT,
    gpu_mode        TEXT DEFAULT 'local',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    started_at      TEXT,
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    epoch       INTEGER NOT NULL,
    file_path   TEXT NOT NULL,
    loss        REAL,
    is_best     INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_settings (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
"""


def _row(row) -> dict[str, Any]:
    return dict(row)


@asynccontextmanager
async def _db():
    """Open a fresh aiosqlite connection for one operation block."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = sqlite3.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        yield db


async def init_db() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.executescript(_SCHEMA_SQL)
        await db.commit()


# ── Projects ───────────────────────────────────────────────────────────────────

async def create_project(
    project_id: str,
    name: str,
    lora_type: str,
    trigger_word: str,
    base_model: str = "",
    config_json: Optional[str] = None,
    gpu_mode: str = "local",
) -> dict[str, Any]:
    async with _db() as db:
        await db.execute(
            """INSERT INTO projects (id, name, lora_type, trigger_word, base_model, config_json, gpu_mode)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (project_id, name, lora_type, trigger_word, base_model, config_json, gpu_mode),
        )
        await db.commit()
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        return _row(await cursor.fetchone())


async def get_project(project_id: str) -> Optional[dict[str, Any]]:
    async with _db() as db:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        return _row(row) if row else None


async def list_projects() -> list[dict[str, Any]]:
    async with _db() as db:
        cursor = await db.execute("SELECT * FROM projects ORDER BY created_at DESC")
        return [_row(r) for r in await cursor.fetchall()]


async def update_project_status(
    project_id: str,
    status: str,
    *,
    error_message: Optional[str] = None,
    lora_file_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    image_count: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    fields = ["status = ?"]
    values: list[Any] = [status]
    for col, val in [
        ("error_message", error_message),
        ("lora_file_path", lora_file_path),
        ("output_dir", output_dir),
        ("started_at", started_at),
        ("completed_at", completed_at),
        ("image_count", image_count),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            values.append(val)
    values.append(project_id)
    async with _db() as db:
        await db.execute(f"UPDATE projects SET {', '.join(fields)} WHERE id = ?", values)
        await db.commit()
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        return _row(row) if row else None


async def update_project_progress(
    project_id: str,
    *,
    current_epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
    current_loss: Optional[float] = None,
    best_loss: Optional[float] = None,
    best_epoch: Optional[int] = None,
    eta_seconds: Optional[int] = None,
) -> None:
    fields, values = [], []
    for col, val in [
        ("current_epoch", current_epoch), ("total_epochs", total_epochs),
        ("current_step", current_step), ("total_steps", total_steps),
        ("current_loss", current_loss), ("best_loss", best_loss),
        ("best_epoch", best_epoch), ("eta_seconds", eta_seconds),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            values.append(val)
    if not fields:
        return
    values.append(project_id)
    async with _db() as db:
        await db.execute(f"UPDATE projects SET {', '.join(fields)} WHERE id = ?", values)
        await db.commit()


async def delete_project(project_id: str) -> bool:
    async with _db() as db:
        cursor = await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        await db.commit()
        return cursor.rowcount > 0


# ── Checkpoints ────────────────────────────────────────────────────────────────

async def upsert_checkpoint(
    project_id: str,
    epoch: int,
    file_path: str,
    loss: Optional[float] = None,
    is_best: bool = False,
) -> dict[str, Any]:
    chk_id = f"{project_id}_ep{epoch:04d}"
    async with _db() as db:
        await db.execute("""
            INSERT INTO checkpoints (id, project_id, epoch, file_path, loss, is_best)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                file_path=excluded.file_path,
                loss=excluded.loss,
                is_best=excluded.is_best
        """, (chk_id, project_id, epoch, file_path, loss, 1 if is_best else 0))
        if is_best:
            await db.execute(
                "UPDATE checkpoints SET is_best=0 WHERE project_id=? AND id!=?",
                (project_id, chk_id),
            )
        await db.commit()
        cursor = await db.execute("SELECT * FROM checkpoints WHERE id=?", (chk_id,))
        return _row(await cursor.fetchone())


async def list_checkpoints(project_id: str) -> list[dict[str, Any]]:
    async with _db() as db:
        cursor = await db.execute(
            "SELECT * FROM checkpoints WHERE project_id=? ORDER BY epoch ASC",
            (project_id,),
        )
        return [_row(r) for r in await cursor.fetchall()]


async def get_checkpoint(project_id: str, epoch: int) -> Optional[dict[str, Any]]:
    chk_id = f"{project_id}_ep{epoch:04d}"
    async with _db() as db:
        cursor = await db.execute("SELECT * FROM checkpoints WHERE id=?", (chk_id,))
        row = await cursor.fetchone()
        return _row(row) if row else None


# ── App Settings ───────────────────────────────────────────────────────────────

async def get_setting(key: str) -> Optional[str]:
    async with _db() as db:
        cursor = await db.execute("SELECT value FROM app_settings WHERE key=?", (key,))
        row = await cursor.fetchone()
        return row["value"] if row else None


async def set_setting(key: str, value: str) -> None:
    async with _db() as db:
        await db.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        await db.commit()

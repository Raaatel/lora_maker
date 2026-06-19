"""Training manager - orchestrates the full pipeline."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from server.services import preprocess_runner, caption_runner, training_runner
from server.services.checkpoint_manager import register_checkpoint_from_event
from server.services.websocket_manager import ConnectionManager
from server import database as db

logger = logging.getLogger(__name__)


class TrainingManager:
    def __init__(self, ws_manager: ConnectionManager) -> None:
        self._ws = ws_manager
        self._lock = asyncio.Lock()
        self._cancel_event: Optional[threading.Event] = None
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()
        self._current_project_id: Optional[str] = None
        self._is_training = False
        self._is_paused = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def current_project_id(self) -> Optional[str]:
        return self._current_project_id

    @property
    def is_training(self) -> bool:
        return self._is_training

    async def start_training(self, project_id: str, config: dict[str, Any]) -> None:
        if self._lock.locked():
            raise RuntimeError("다른 학습이 진행 중입니다")

        async with self._lock:
            self._current_project_id = project_id
            self._is_training = True
            self._cancel_event = threading.Event()
            self._pause_event.set()
            self._is_paused = False
            self._loop = asyncio.get_running_loop()

            try:
                await db.update_project_status(
                    project_id, "running",
                    started_at=datetime.now(timezone.utc).isoformat(),
                )
                await self._ws.broadcast(project_id, {"type": "status", "status": "running"})
                await asyncio.to_thread(self._pipeline, project_id, config, self._cancel_event)

            except Exception as exc:
                error_msg = str(exc)
                logger.exception("Pipeline failed for project %s", project_id)
                await db.update_project_status(
                    project_id, "failed",
                    error_message=error_msg,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                await self._ws.broadcast(project_id, {"type": "error", "message": error_msg})
            finally:
                self._is_training = False
                self._is_paused = False
                self._pause_event.set()
                self._current_project_id = None
                self._cancel_event = None
                self._loop = None

    def pause_training(self) -> bool:
        if not self._is_training:
            return False
        self._pause_event.clear()
        self._is_paused = True
        return True

    def resume_training(self) -> bool:
        if not self._is_paused:
            return False
        self._pause_event.set()
        self._is_paused = False
        return True

    async def cancel_training(self) -> bool:
        if self._cancel_event is None:
            return False
        if self._is_paused:
            self._pause_event.set()
            self._is_paused = False
        self._cancel_event.set()
        return True

    async def cancel_all(self) -> None:
        await self.cancel_training()

    def _pipeline(self, project_id: str, config: dict[str, Any], cancel_event: threading.Event) -> None:
        def make_callback(stage: str):
            def _cb(event_type: str, data: dict[str, Any]) -> None:
                self._run(self._handle_event(project_id, stage, event_type, data))
            return _cb

        # 1) Preprocess
        if cancel_event.is_set():
            return
        skip = config.get("preprocess", {}).get("skip_preprocess", False)
        if skip:
            # Copy raw → processed as-is
            import shutil
            from pathlib import Path
            raw = Path(config["data"]["raw_dir"])
            proc = Path(config["data"]["processed_dir"])
            proc.mkdir(parents=True, exist_ok=True)
            for f in raw.iterdir():
                if f.is_file():
                    shutil.copy2(str(f), str(proc / f.name))
            make_callback("preprocess")("preprocess_complete", {"processed": len(list(proc.iterdir()))})
        else:
            self._run(db.update_project_status(project_id, "preprocessing"))
            self._run(self._ws.broadcast(project_id, {"type": "status", "status": "preprocessing"}))
            preprocess_runner.run(config, make_callback("preprocess"))

        # 2) Caption
        if cancel_event.is_set():
            return
        self._run(db.update_project_status(project_id, "captioning"))
        self._run(self._ws.broadcast(project_id, {"type": "status", "status": "captioning"}))
        caption_runner.run(config, make_callback("caption"))

        # 3) Train
        if cancel_event.is_set():
            return
        self._run(db.update_project_status(project_id, "training"))
        self._run(self._ws.broadcast(project_id, {"type": "status", "status": "training"}))
        result = training_runner.run(config, make_callback("training"), cancel_event, self._pause_event)

        # 4) Finalize
        if result.get("cancelled"):
            self._run(db.update_project_status(
                project_id, "cancelled",
                completed_at=datetime.now(timezone.utc).isoformat(),
            ))
            self._run(self._ws.broadcast(project_id, {"type": "status", "status": "cancelled"}))
        elif result.get("error"):
            self._run(db.update_project_status(
                project_id, "failed",
                error_message=result["error"],
                completed_at=datetime.now(timezone.utc).isoformat(),
            ))
            self._run(self._ws.broadcast(project_id, {"type": "status", "status": "failed", "error": result["error"]}))
        else:
            lora_path = result.get("lora_path", "")
            self._run(db.update_project_status(
                project_id, "completed",
                lora_file_path=lora_path,
                completed_at=datetime.now(timezone.utc).isoformat(),
            ))
            self._run(self._ws.broadcast(project_id, {
                "type": "status", "status": "completed", "lora_path": lora_path,
            }))

    async def _handle_event(self, project_id: str, stage: str, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "step":
            await db.update_project_progress(
                project_id,
                current_epoch=data.get("epoch"),
                total_epochs=data.get("total_epochs"),
                current_step=data.get("step"),
                total_steps=data.get("total_steps"),
                current_loss=data.get("loss"),
                eta_seconds=data.get("eta_seconds"),
            )
            await self._ws.broadcast(project_id, {"type": "progress", **data})

        elif event_type == "epoch_end":
            loss = data.get("avg_loss")
            epoch = data.get("epoch")
            await db.update_project_progress(
                project_id,
                current_epoch=epoch,
                total_epochs=data.get("total_epochs"),
                current_loss=loss,
            )
            await self._ws.broadcast(project_id, {"type": "epoch_end", **data})

        elif event_type == "checkpoint_saved":
            epoch = data.get("epoch")
            file_path = data.get("file_path")
            if epoch is not None and file_path:
                project = await db.get_project(project_id)
                loss = project.get("current_loss") if project else None
                chk = await register_checkpoint_from_event(project_id, epoch, file_path, loss)
                await self._ws.broadcast(project_id, {"type": "checkpoint_saved", "checkpoint": chk})

        elif event_type == "completed":
            await self._ws.broadcast(project_id, {"type": "training_complete", **data})

        elif event_type == "error":
            await db.update_project_status(
                project_id, "failed",
                error_message=data.get("message", "Unknown error"),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            await self._ws.broadcast(project_id, {"type": "error", **data})

        else:
            await self._ws.broadcast(project_id, {"type": event_type, "stage": stage, **data})

    def _run(self, coro) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            future.result(timeout=30)
        except Exception:
            logger.exception("Failed to run coroutine from thread")

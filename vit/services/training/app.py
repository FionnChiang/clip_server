import os
import json
import logging
import uuid
import tempfile
import shutil
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR = os.environ.get("MODELS_DIR", "/shared/models")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/shared/output")
TEMP_DIR = os.environ.get("TEMP_DIR", "/shared/tmp")

_jobs: dict[str, dict] = {}
_ws_connections: dict[str, list[WebSocket]] = {}

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class TrainingConfigPayload(BaseModel):
    batch_size: int = 32
    epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 0.0001
    lr_scheduler: str = "cosine"
    early_stop_patience: int = 10
    class_balance: str = "weighted_loss"
    num_workers: int = 4
    save_best_only: bool = True
    mixed_precision: bool = False
    log_interval: int = 10


class ModelConfigPayload(BaseModel):
    path: str = "/shared/models"
    freeze_encoder: bool = True
    dropout: float = 0.1
    projection_dim: Optional[int] = None
    pool: str = "cls"


class DataManifestItem(BaseModel):
    url: str
    category: str
    split: str = "train"
    filename: str = ""


class StartTrainingRequest(BaseModel):
    config: dict
    categories: list[str]
    data_manifest: list[DataManifestItem]
    train_ratio: float = 0.8
    seed: int = 42


class JobStatus(BaseModel):
    id: str
    status: str
    current_epoch: int
    total_epochs: int
    best_val_acc: float
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Training Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
async def health():
    import torch
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "models_dir": MODELS_DIR,
        "output_dir": OUTPUT_DIR,
    }


@app.post("/api/train", response_model=JobStatus)
async def start_training(req: StartTrainingRequest):
    job_id = uuid.uuid4().hex
    total_epochs = req.config.get("training", {}).get("epochs", 50)

    job_path = _resolve_output(job_id)
    Path(job_path).mkdir(parents=True, exist_ok=True)

    job_rec = {
        "id": job_id,
        "status": "preparing",
        "current_epoch": 0,
        "total_epochs": total_epochs,
        "best_val_acc": 0.0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }
    _jobs[job_id] = job_rec

    # Download images from S3 presigned URLs
    data_root = await _download_and_organize(job_id, req.data_manifest)

    # Run training in background thread
    asyncio.create_task(
        _run_training(
            job_id,
            req.categories,
            req.config,
            data_root,
            job_path,
            req.train_ratio,
            req.seed,
        )
    )

    job_rec["status"] = "running"

    return JobStatus(
        id=job_id,
        status="running",
        current_epoch=0,
        total_epochs=total_epochs,
        best_val_acc=0.0,
        started_at=job_rec["started_at"],
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)


@app.get("/api/jobs/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    metrics_path = Path(_resolve_output(job_id)) / "training_history.json"
    if not metrics_path.exists():
        return []
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["_stop_requested"] = True
    return {"ok": True}


@app.websocket("/ws/training/{job_id}")
async def ws_training(ws: WebSocket, job_id: str):
    await ws.accept()
    if job_id not in _ws_connections:
        _ws_connections[job_id] = []
    _ws_connections[job_id].append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_connections[job_id].remove(ws)
        if not _ws_connections[job_id]:
            del _ws_connections[job_id]
    except Exception:
        if job_id in _ws_connections and ws in _ws_connections[job_id]:
            _ws_connections[job_id].remove(ws)


async def _broadcast(job_id: str, msg: dict):
    if job_id in _ws_connections:
        dead = []
        for ws in _ws_connections[job_id]:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_connections[job_id].remove(ws)


def _resolve_output(job_id: str) -> str:
    return str(Path(OUTPUT_DIR) / job_id)


async def _download_and_organize(job_id: str, manifest: list[DataManifestItem]) -> Path:
    base = Path(TEMP_DIR) / job_id / "data"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        sem = asyncio.Semaphore(16)

        async def download_one(item: DataManifestItem):
            async with sem:
                cat_dir = base / item.category / item.split
                cat_dir.mkdir(parents=True, exist_ok=True)
                fname = item.filename or f"{uuid.uuid4().hex}.jpg"
                fpath = cat_dir / fname
                try:
                    resp = await client.get(item.url)
                    resp.raise_for_status()
                    fpath.write_bytes(resp.content)
                except Exception as e:
                    logger.warning(f"Failed to download {item.url}: {e}")

        tasks = [download_one(item) for item in manifest]
        await asyncio.gather(*tasks)

    return base


async def _run_training(
    job_id: str,
    categories: list[str],
    config: dict,
    data_root: Path,
    output_dir: str,
    train_ratio: float,
    seed: int,
):
    import sys
    import traceback
    import torch

    # The src/ package is mounted at /app/src (from project root at vit/)
    SRC_DIR = os.environ.get("SRC_DIR", "/app")
    sys.path.insert(0, SRC_DIR)

    try:
        from src.data.dataset import build_dataloaders
        from src.models.classifier import LayoutClassifier
        from src.trainers.trainer import Trainer

        await _broadcast(job_id, {
            "type": "progress",
            "job_id": job_id,
            "data": {"epoch": 0, "status": "preparing", "message": "Initializing model..."}
        })

        model_cfg = config.get("model", {})
        model_path = model_cfg.get("path", MODELS_DIR)
        training_cfg = config.get("training", {})

        train_loader, val_loader, num_classes, class_weights = build_dataloaders(
            data_root=str(data_root),
            categories=categories,
            batch_size=training_cfg.get("batch_size", 32),
            train_ratio=train_ratio,
            seed=seed,
            num_workers=training_cfg.get("num_workers", 4),
            class_balance=training_cfg.get("class_balance", "weighted_loss"),
        )

        model = LayoutClassifier(
            model_path=model_path,
            num_classes=num_classes,
            freeze_encoder=model_cfg.get("freeze_encoder", True),
            dropout=model_cfg.get("dropout", 0.1),
            projection_dim=model_cfg.get("projection_dim"),
            pool=model_cfg.get("pool", "cls"),
        )
        model.set_categories(categories)

        full_config = {
            "data": {"root": str(data_root), "categories": categories, "train_ratio": train_ratio, "seed": seed},
            "model": model_cfg,
            "training": training_cfg,
            "output": {"dir": output_dir, "model_name": f"{job_id}.pth"},
            "calibration": config.get("calibration", {}),
        }

        class CustomTrainer(Trainer):
            def train(self):
                epochs = self.training_cfg["epochs"]
                patience = self.training_cfg.get("early_stop_patience", epochs)
                for epoch in range(1, epochs + 1):
                    job = _jobs.get(job_id)
                    if job and job.get("_stop_requested"):
                        asyncio.run(_broadcast(job_id, {
                            "type": "stopped",
                            "job_id": job_id,
                            "data": {"message": "Training stopped by user"},
                        }))
                        return

                    train_loss, train_acc = self._run_epoch(epoch, is_train=True)
                    val_loss, val_acc = self._run_epoch(epoch, is_train=False)

                    if self.scheduler is not None:
                        self.scheduler.step()
                    lr = self.optimizer.param_groups[0]["lr"]

                    asyncio.run(_broadcast(job_id, {
                        "type": "progress",
                        "job_id": job_id,
                        "data": {
                            "epoch": epoch,
                            "train_loss": round(train_loss, 6),
                            "train_acc": round(train_acc, 6),
                            "val_loss": round(val_loss, 6),
                            "val_acc": round(val_acc, 6),
                            "lr": round(lr, 8),
                            "status": "running",
                        }
                    }))

                    record = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                              "val_loss": val_loss, "val_acc": val_acc, "lr": lr}
                    self.history.append(record)

                    self._maybe_save_checkpoint(val_acc, epoch)

                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_epoch = epoch
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= patience:
                        asyncio.run(_broadcast(job_id, {
                            "type": "progress",
                            "job_id": job_id,
                            "data": {"epoch": epoch, "status": "early_stopping",
                                     "message": f"Early stopping. Best val acc: {self.best_val_acc:.4f}"}
                        }))
                        break

                    # Update job status
                    job = _jobs.get(job_id)
                    if job:
                        job["current_epoch"] = epoch
                        if val_acc > job["best_val_acc"]:
                            job["best_val_acc"] = val_acc

                self._save_history()

        trainer = CustomTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=full_config,
            class_weights=class_weights,
        )
        trainer.train()

        # 温度校准 + 置信度拒绝阈值（自动写入 checkpoint 的 calibration 字段）
        calibration = {}
        if config.get("calibration", {}).get("enabled", True):
            calibration = trainer.calibrate() or {}

        # Save checkpoint
        best_ckpt = Path(output_dir) / "best_model.pth"
        asyncio.run(_broadcast(job_id, {
            "type": "completed",
            "job_id": job_id,
            "data": {
                "best_val_acc": trainer.best_val_acc,
                "best_epoch": trainer.best_epoch,
                "checkpoint_path": str(best_ckpt),
                "output_dir": output_dir,
                "calibration": calibration,
            }
        }))

        job = _jobs.get(job_id)
        if job:
            job["status"] = "completed"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        logger.error(f"Training failed: {traceback.format_exc()}")
        asyncio.run(_broadcast(job_id, {
            "type": "error",
            "job_id": job_id,
            "data": {"message": str(e), "traceback": traceback.format_exc()}
        }))
        job = _jobs.get(job_id)
        if job:
            job["status"] = "failed"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()

    finally:
        # Cleanup temp data
        try:
            shutil.rmtree(Path(TEMP_DIR) / job_id / "data", ignore_errors=True)
        except Exception:
            pass

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import select, func

from ..database import get_session, Project, DatasetImage, TrainingJob, TrainingMetric, Model
from ..s3_client import s3_client
from ..config import server_config
from ..websocket import ws_manager
from ..schemas.training import StartTrainingRequest, TrainingJobOut

logger = logging.getLogger(__name__)

_active_jobs: dict[str, asyncio.Task] = {}


def _key_from_url(url: str) -> Optional[str]:
    if url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        if len(parts) == 2:
            return parts[1]
    return None


async def start_training(project_id: str, request: StartTrainingRequest) -> TrainingJobOut:
    async with get_session() as session:
        project_result = await session.execute(select(Project).where(Project.id == project_id))
        project = project_result.scalar()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        if not project.categories:
            raise ValueError("Project has no categories. Please upload images first.")

        total = await session.execute(
            select(func.count(DatasetImage.id)).where(DatasetImage.project_id == project_id)
        )
        if (total.scalar() or 0) == 0:
            raise ValueError("No images in project. Please upload images first.")

        model_cfg = request.model.model_dump()
        training_cfg = request.training.model_dump()
        config = {
            "model": model_cfg,
            "training": training_cfg,
            "seed": request.seed,
            "train_ratio": request.train_ratio,
        }

        images_result = await session.execute(
            select(DatasetImage).where(DatasetImage.project_id == project_id)
        )
        images = images_result.scalars().all()

        presigned_expiry = 86400
        data_manifest = []
        for img in images:
            key = _key_from_url(img.s3_url)
            if not key:
                continue
            try:
                presigned = s3_client.get_presigned_url(key, expires=presigned_expiry)
            except Exception as e:
                logger.warning(f"Failed to generate presigned URL for {img.s3_url}: {e}")
                continue
            data_manifest.append({
                "url": presigned,
                "category": img.category,
                "split": img.split or "unassigned",
                "filename": img.original_filename or f"{img.id}.jpg",
            })

        if not data_manifest:
            raise ValueError("No valid images found in project")

        total_epochs_value = training_cfg.get("epochs", 50)

        job = TrainingJob(
            project_id=project_id,
            status="pending",
            config=config,
            total_epochs=total_epochs_value,
            started_at=datetime.now(timezone.utc),
        )
        session.add(job)
        await session.flush()
        job_id = job.id

    training_url = server_config.training_url

    async def _send_training_request():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                payload = {
                    "config": config,
                    "categories": list(project.categories),
                    "data_manifest": data_manifest,
                    "train_ratio": request.train_ratio,
                    "seed": request.seed,
                }
                resp = await client.post(f"{training_url}/api/train", json=payload)
                resp.raise_for_status()
                result = resp.json()
                training_job_id = result["id"]

            async with get_session() as session:
                result_job = await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
                db_job = result_job.scalar()
                if db_job:
                    db_job.status = "running"

            asyncio.create_task(_monitor_training(job_id, training_job_id, training_url))

        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            async with get_session() as session:
                result_job = await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
                db_job = result_job.scalar()
                if db_job:
                    db_job.status = "failed"
                    db_job.finished_at = datetime.now(timezone.utc)
            await ws_manager.broadcast_to_all({
                "type": "error",
                "job_id": job_id,
                "data": {"message": str(e)},
            })

    task = asyncio.create_task(_send_training_request())
    _active_jobs[job_id] = task
    job.status = "running"

    return TrainingJobOut(
        id=job.id,
        project_id=job.project_id,
        status="running",
        current_epoch=0,
        total_epochs=total_epochs_value,
        best_val_acc=0.0,
        started_at=job.started_at,
        finished_at=None,
    )


async def _monitor_training(backend_job_id: str, training_job_id: str, training_url: str):
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            while True:
                try:
                    resp = await client.get(f"{training_url}/api/jobs/{training_job_id}")
                    resp.raise_for_status()
                    job_data = resp.json()

                    async with get_session() as session:
                        result = await session.execute(
                            select(TrainingJob).where(TrainingJob.id == backend_job_id)
                        )
                        db_job = result.scalar()
                        if db_job:
                            db_job.status = job_data.get("status", db_job.status)
                            db_job.current_epoch = job_data.get("current_epoch", 0)
                            db_job.best_val_acc = job_data.get("best_val_acc", db_job.best_val_acc)
                            if job_data.get("status") in ("completed", "failed", "stopped"):
                                db_job.finished_at = datetime.now(timezone.utc)

                    if job_data.get("status") in ("completed", "failed", "stopped"):
                        if job_data.get("status") == "completed":
                            metrics_resp = await client.get(f"{training_url}/api/jobs/{training_job_id}/metrics")
                            metrics_data = metrics_resp.json() if metrics_resp.status_code == 200 else []
                            async with get_session() as session:
                                for m in metrics_data:
                                    metric = TrainingMetric(
                                        job_id=backend_job_id,
                                        epoch=m.get("epoch", 0),
                                        train_loss=m.get("train_loss"),
                                        train_acc=m.get("train_acc"),
                                        val_loss=m.get("val_loss"),
                                        val_acc=m.get("val_acc"),
                                        lr=m.get("lr"),
                                    )
                                    session.add(metric)

                            # Populate metrics in broadcast message
                            await ws_manager.broadcast_to_all({
                                "type": "completed",
                                "job_id": backend_job_id,
                                "data": {
                                    "best_val_acc": job_data.get("best_val_acc", 0),
                                    "message": f"Training completed with best val_acc: {job_data.get('best_val_acc', 0):.4f}",
                                }
                            })

                            # Create model record
                            async with get_session() as session:
                                model_rec = Model(
                                    project_id=db_job.project_id if db_job else "",
                                    job_id=backend_job_id,
                                    name=f"model_{backend_job_id[:8]}",
                                    checkpoint_path=f"/shared/output/{training_job_id}/best_model.pth",
                                    val_acc=job_data.get("best_val_acc", 0),
                                )
                                session.add(model_rec)

                            break

                        elif job_data.get("status") == "failed":
                            await ws_manager.broadcast_to_all({
                                "type": "error",
                                "job_id": backend_job_id,
                                "data": {"message": "Training failed in training service"},
                            })
                            break
                        elif job_data.get("status") == "stopped":
                            await ws_manager.broadcast_to_all({
                                "type": "stopped",
                                "job_id": backend_job_id,
                                "data": {"message": "Training stopped by user"},
                            })
                            break

                except Exception as e:
                    logger.warning(f"Training monitor error: {e}")

                await asyncio.sleep(3)

    except Exception as e:
        logger.error(f"Training monitor failed: {e}")

    finally:
        if backend_job_id in _active_jobs:
            del _active_jobs[backend_job_id]


async def stop_training(job_id: str) -> bool:
    async with get_session() as session:
        result = await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar()
        if not job:
            return False

    training_url = server_config.training_url
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            await client.post(f"{training_url}/api/jobs/{job_id}/stop")
    except Exception as e:
        logger.warning(f"Failed to stop training service job: {e}")

    async with get_session() as session:
        result = await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar()
        if job:
            job.status = "stopped"
            job.finished_at = datetime.now(timezone.utc)

    await ws_manager.broadcast_to_all({
        "type": "stopped",
        "job_id": job_id,
        "data": {"message": "Training stopped by user"},
    })

    if job_id in _active_jobs:
        task = _active_jobs.pop(job_id)
        task.cancel()
    return True


async def list_jobs(project_id: str) -> list[TrainingJobOut]:
    async with get_session() as session:
        result = await session.execute(
            select(TrainingJob).where(TrainingJob.project_id == project_id).order_by(TrainingJob.started_at.desc())
        )
        jobs = result.scalars().all()
        return [
            TrainingJobOut(
                id=j.id,
                project_id=j.project_id,
                status=j.status,
                current_epoch=j.current_epoch or 0,
                total_epochs=j.total_epochs or 0,
                best_val_acc=j.best_val_acc or 0.0,
                started_at=j.started_at,
                finished_at=j.finished_at,
            )
            for j in jobs
        ]


async def get_job(job_id: str) -> Optional[TrainingJob]:
    async with get_session() as session:
        result = await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        return result.scalar()


async def get_job_metrics(job_id: str) -> list:
    async with get_session() as session:
        result = await session.execute(
            select(TrainingMetric).where(TrainingMetric.job_id == job_id).order_by(TrainingMetric.epoch)
        )
        metrics = result.scalars().all()
        return [
            {
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "train_acc": m.train_acc,
                "val_loss": m.val_loss,
                "val_acc": m.val_acc,
                "lr": m.lr,
            }
            for m in metrics
        ]

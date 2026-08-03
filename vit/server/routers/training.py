from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..schemas.training import (
    StartTrainingRequest, TrainingJobOut, TrainingMetricOut,
)
from ..services import training_client
from ..websocket import ws_manager

router = APIRouter(prefix="/api/projects/{project_id}", tags=["training"])


@router.get("/jobs", response_model=list[TrainingJobOut])
async def list_jobs(project_id: str):
    return await training_client.list_jobs(project_id)


@router.post("/train", response_model=TrainingJobOut)
async def start_training(project_id: str, request: StartTrainingRequest):
    try:
        return await training_client.start_training(project_id, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}", response_model=TrainingJobOut)
async def get_job(project_id: str, job_id: str):
    job = await training_client.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return TrainingJobOut(
        id=job.id,
        project_id=job.project_id,
        status=job.status,
        current_epoch=job.current_epoch or 0,
        total_epochs=job.total_epochs or 0,
        best_val_acc=job.best_val_acc or 0.0,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.post("/jobs/{job_id}/stop")
async def stop_training(project_id: str, job_id: str):
    if not await training_client.stop_training(job_id):
        raise HTTPException(status_code=404, detail="Job not found or already finished")
    return {"ok": True}


@router.get("/jobs/{job_id}/metrics")
async def get_job_metrics(project_id: str, job_id: str):
    return await training_client.get_job_metrics(job_id)


@router.websocket("/ws/training/{job_id}")
async def ws_training(job_id: str, ws: WebSocket):
    await ws_manager.connect(job_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, ws)
    except Exception:
        ws_manager.disconnect(job_id, ws)

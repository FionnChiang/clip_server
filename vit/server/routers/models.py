from fastapi import APIRouter, HTTPException

from ..services import model_service

router = APIRouter(prefix="/api/projects/{project_id}", tags=["models"])


@router.get("/models")
async def list_models(project_id: str):
    return await model_service.list_models(project_id)


@router.delete("/models/{model_id}")
async def delete_model(project_id: str, model_id: str):
    if not await model_service.delete_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"ok": True}

import io
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from PIL import Image

from ..schemas.inference import PredictionResult, BatchPredictionResult
from ..services import inference_client

router = APIRouter(prefix="/api/projects/{project_id}", tags=["inference"])


@router.post("/predict")
async def predict(
    project_id: str,
    file: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
):
    contents = await file.read()
    try:
        results = await inference_client.predict(project_id, contents, model_id, top_k=1)
        return results[0] if results else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/predict/top-k")
async def predict_top_k(
    project_id: str,
    file: UploadFile = File(...),
    k: int = Query(3, ge=1, le=10),
    model_id: Optional[str] = Form(None),
):
    contents = await file.read()
    try:
        results = await inference_client.predict(project_id, contents, model_id, top_k=k)
        return {"results": [r.model_dump() for r in results]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

import os
import io
import base64
import logging
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/shared/output")
SRC_DIR = os.environ.get("SRC_DIR", "/app")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/shared/output/checkpoint.pth")

import sys
sys.path.insert(0, SRC_DIR)

app = FastAPI(title="Inference Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_predictor = None
_loaded_checkpoint = None


class PredictRequest(BaseModel):
    image_base64: str
    checkpoint_path: str = ""


class PredictionResult(BaseModel):
    category: str
    index: int
    confidence: float
    probabilities: dict[str, float]


def _get_predictor(checkpoint_path: str):
    global _predictor, _loaded_checkpoint
    ckpt = checkpoint_path or CHECKPOINT_PATH
    if _predictor is not None and _loaded_checkpoint == ckpt:
        return _predictor
    from src.inference.predictor import LayoutPredictor
    _predictor = LayoutPredictor(ckpt, device="cpu")
    _loaded_checkpoint = ckpt
    logger.info(f"Loaded model from {ckpt}, categories: {_predictor.categories}")
    return _predictor


@app.get("/api/health")
async def health():
    import torch
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_path": CHECKPOINT_PATH,
    }


@app.get("/api/categories")
async def categories():
    predictor = _get_predictor("")
    return {"categories": predictor.categories}


@app.post("/api/predict", response_model=PredictionResult)
async def predict(req: PredictRequest):
    try:
        predictor = _get_predictor(req.checkpoint_path)
        image_data = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        result = predictor.predict(image)
        return PredictionResult(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/top-k")
async def predict_top_k(req: PredictRequest, k: int = 3):
    try:
        predictor = _get_predictor(req.checkpoint_path)
        image_data = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        results = predictor.predict_top_k(image, k=k)
        probs = predictor.predict(image)
        return {
            "results": [
                {**r, "probabilities": probs.get("probabilities", {})}
                for r in results
            ]
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch")
async def predict_batch(req: list[PredictRequest]):
    try:
        predictor = _get_predictor(req[0].checkpoint_path if req else "")
        results = []
        for r in req:
            image_data = base64.b64decode(r.image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            result = predictor.predict(image)
            results.append(result)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

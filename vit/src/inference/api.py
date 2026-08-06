import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from .predictor import LayoutPredictor


class PredictionResult(BaseModel):
    category: str
    index: int
    confidence: float
    probabilities: dict[str, float]
    rejected: bool = False
    reason: Optional[str] = None
    original_category: Optional[str] = None
    original_index: Optional[int] = None


class TopKResult(BaseModel):
    results: list[dict]


predictor: Optional[LayoutPredictor] = None


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def create_app(
    checkpoint_path: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
    temperature: Optional[float] = None,
) -> FastAPI:
    global predictor

    if checkpoint_path is None:
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", "output/best_model.pth")
    confidence_threshold = _env_float("CONFIDENCE_THRESHOLD", confidence_threshold)
    margin_threshold = _env_float("MARGIN_THRESHOLD", margin_threshold)
    temperature = _env_float("TEMPERATURE", temperature)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global predictor
        predictor = LayoutPredictor(
            checkpoint_path,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            temperature=temperature,
        )
        print(f"Model loaded. Categories: {predictor.categories}")
        print(f"Device: {predictor.device}")
        print(f"Calibration: T={predictor.temperature}, "
              f"conf_th={predictor.confidence_threshold}, margin_th={predictor.margin_threshold}")
        yield

    app = FastAPI(
        title="Layout Classifier API",
        description="CLIP ViT 版式分类推理服务",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "device": str(predictor.device)}

    @app.get("/categories")
    async def get_categories():
        return {"categories": predictor.categories}

    @app.post("/predict", response_model=PredictionResult)
    async def predict_image(file: UploadFile = File(...)):
        if file.content_type not in ("image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"):
            raise HTTPException(status_code=400, detail="Unsupported image format")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predictor.predict(image)
        return result

    @app.post("/predict/top-k")
    async def predict_top_k(file: UploadFile = File(...), k: int = 3):
        if file.content_type not in ("image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"):
            raise HTTPException(status_code=400, detail="Unsupported image format")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = predictor.predict_top_k(image, k=k)
        return {"results": results}

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )

    return app


def serve(
    checkpoint_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    confidence_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
    temperature: Optional[float] = None,
):
    app = create_app(
        checkpoint_path,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        temperature=temperature,
    )
    uvicorn.run(app, host=host, port=port)

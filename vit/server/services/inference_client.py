import io
import base64
import logging
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import select

from ..database import get_session, Model
from ..config import server_config
from ..schemas.inference import PredictionResult

logger = logging.getLogger(__name__)


async def predict(
    project_id: str,
    image_bytes: bytes,
    model_id: Optional[str] = None,
    top_k: int = 1,
    confidence_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
    temperature: Optional[float] = None,
) -> list[PredictionResult]:
    async with get_session() as session:
        if model_id:
            result = await session.execute(select(Model).where(Model.id == model_id))
            model_rec = result.scalar()
        else:
            result = await session.execute(
                select(Model)
                .where(Model.project_id == project_id, Model.status == "available")
                .order_by(Model.val_acc.desc())
                .limit(1)
            )
            model_rec = result.scalar()

        if not model_rec:
            raise ValueError("No trained model found for this project")

        ckpt_path = model_rec.checkpoint_path

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    inference_url = server_config.inference_url
    payload = {
        "image_base64": image_b64,
        "checkpoint_path": ckpt_path,
    }
    # 后端全局默认覆盖（server_config.yaml 的 inference 段）
    if confidence_threshold is None:
        confidence_threshold = server_config.inference_confidence_threshold
    if margin_threshold is None:
        margin_threshold = server_config.inference_margin_threshold
    if temperature is None:
        temperature = server_config.inference_temperature
    if confidence_threshold is not None:
        payload["confidence_threshold"] = confidence_threshold
    if margin_threshold is not None:
        payload["margin_threshold"] = margin_threshold
    if temperature is not None:
        payload["temperature"] = temperature

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        if top_k <= 1:
            resp = await client.post(f"{inference_url}/api/predict", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return [PredictionResult(**data)]
        else:
            resp = await client.post(f"{inference_url}/api/predict/top-k", json=payload, params={"k": top_k})
            resp.raise_for_status()
            data = resp.json()
            return [
                PredictionResult(**r)
                for r in data.get("results", [])
            ]


async def predict_pages(
    project_id: str,
    page_images: list[tuple[int, bytes]],
    model_id: Optional[str] = None,
    top_k: int = 1,
    confidence_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
    temperature: Optional[float] = None,
) -> list[tuple[int, list[PredictionResult]]]:
    """对文档的每一页图片逐页预测。

    Args:
        page_images: [(page_no, image_bytes), ...]，page_no 从 1 开始。

    Returns:
        [(page_no, [PredictionResult, ...]), ...]
    """
    results = []
    for page_no, image_bytes in page_images:
        page_results = await predict(
            project_id, image_bytes, model_id, top_k=top_k,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            temperature=temperature,
        )
        results.append((page_no, page_results))
    return results

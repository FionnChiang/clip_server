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

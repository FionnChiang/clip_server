import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from ..database import get_session, Model

logger = logging.getLogger(__name__)


async def list_models(project_id: str) -> list[dict]:
    async with get_session() as session:
        result = await session.execute(
            select(Model)
            .where(Model.project_id == project_id, Model.status != "deleted")
            .order_by(Model.created_at.desc())
        )
        models = result.scalars().all()
        return [
            {
                "id": m.id,
                "project_id": m.project_id,
                "job_id": m.job_id,
                "name": m.name,
                "checkpoint_path": m.checkpoint_path,
                "val_acc": m.val_acc,
                "categories": m.categories or [],
                "config": m.config or {},
                "status": m.status,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in models
        ]


async def delete_model(model_id: str) -> bool:
    async with get_session() as session:
        result = await session.execute(select(Model).where(Model.id == model_id))
        model = result.scalar()
        if not model:
            return False
        model.status = "deleted"
        return True

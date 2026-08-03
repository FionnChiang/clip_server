import logging

from sqlalchemy import select, func, delete
from sqlalchemy.orm import selectinload

from ..database import get_session, Project, DatasetImage, TrainingJob, TrainingMetric, Model, InferenceService
from ..schemas.project import ProjectCreate, ProjectUpdate, ProjectOut

logger = logging.getLogger(__name__)


async def list_projects() -> list[ProjectOut]:
    async with get_session() as session:
        result = await session.execute(select(Project).order_by(Project.updated_at.desc()))
        projects = result.scalars().all()
        out = []
        for p in projects:
            img_count_result = await session.execute(
                select(func.count(DatasetImage.id)).where(DatasetImage.project_id == p.id)
            )
            img_count = img_count_result.scalar() or 0
            model_count_result = await session.execute(
                select(func.count(Model.id)).where(Model.project_id == p.id)
            )
            model_count = model_count_result.scalar() or 0
            latest_job_result = await session.execute(
                select(TrainingJob).where(TrainingJob.project_id == p.id).order_by(TrainingJob.started_at.desc()).limit(1)
            )
            latest_job = latest_job_result.scalar()
            po = ProjectOut(
                id=p.id,
                name=p.name,
                description=p.description or "",
                model_path=p.model_path,
                categories=p.categories or [],
                config=p.config or {},
                status=p.status,
                created_at=p.created_at,
                updated_at=p.updated_at,
                image_count=img_count,
                model_count=model_count,
                latest_job_status=latest_job.status if latest_job else None,
            )
            out.append(po)
        return out


async def create_project(data: ProjectCreate) -> Project:
    async with get_session() as session:
        project = Project(
            name=data.name,
            description=data.description or "",
            model_path=data.model_path,
            categories=[],
            config={},
        )
        session.add(project)
        await session.flush()
        return project


async def get_project(project_id: str) -> Project | None:
    async with get_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_id))
        return result.scalar()


async def update_project(project_id: str, data: ProjectUpdate) -> Project | None:
    async with get_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar()
        if not project:
            return None
        update_data = data.model_dump(exclude_unset=True)
        for k, v in update_data.items():
            setattr(project, k, v)
        return project


async def delete_project(project_id: str) -> bool:
    async with get_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar()
        if not project:
            return False
        await session.delete(project)
        return True

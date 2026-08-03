from fastapi import APIRouter, HTTPException

from ..schemas.project import ProjectCreate, ProjectUpdate, ProjectOut
from ..services import project_service

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[ProjectOut])
async def list_projects():
    return await project_service.list_projects()


@router.post("", response_model=ProjectOut)
async def create_project(data: ProjectCreate):
    project = await project_service.create_project(data)
    return ProjectOut(
        id=project.id,
        name=project.name,
        description=project.description or "",
        model_path=project.model_path,
        categories=project.categories or [],
        config=project.config or {},
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
        image_count=0,
        model_count=0,
        latest_job_status=None,
    )


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(project_id: str):
    projects_list = await project_service.list_projects()
    for p in projects_list:
        if p.id == project_id:
            return p
    raise HTTPException(status_code=404, detail="Project not found")


@router.put("/{project_id}", response_model=ProjectOut)
async def update_project(project_id: str, data: ProjectUpdate):
    project = await project_service.update_project(project_id, data)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    projects_list = await project_service.list_projects()
    for p in projects_list:
        if p.id == project_id:
            return p
    raise HTTPException(status_code=404, detail="Project not found")


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    if not await project_service.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True}

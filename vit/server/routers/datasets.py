from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query

from ..schemas.dataset import (
    CategoryOut, CategoryCreate, PaginatedImages,
    SplitRequest, UpdateImageCategory, UpdateImageSplit,
)
from ..services import dataset_service

router = APIRouter(prefix="/api/projects/{project_id}", tags=["datasets"])


@router.get("/categories", response_model=list[CategoryOut])
async def list_categories(project_id: str):
    return await dataset_service.list_categories(project_id)


@router.post("/categories")
async def add_category(project_id: str, data: CategoryCreate):
    await dataset_service.add_category(project_id, data.name)
    return {"ok": True}


@router.delete("/categories/{name}")
async def remove_category(project_id: str, name: str):
    deleted = await dataset_service.remove_category(project_id, name)
    return {"ok": True, "deleted_images": deleted}


@router.post("/upload")
async def upload_images(
    project_id: str,
    files: list[UploadFile] = File(...),
    category: str = Form(...),
):
    file_data = []
    for f in files:
        contents = await f.read()
        file_data.append((f.filename, contents, f.content_type or "image/jpeg"))

    try:
        results = await dataset_service.upload_images(project_id, category, file_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "uploaded": len(results), "images": results}


@router.get("/images", response_model=PaginatedImages)
async def list_images(
    project_id: str,
    category: str = Query(None),
    split: str = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    return await dataset_service.list_images(
        project_id, category=category, split=split, page=page, page_size=page_size,
    )


@router.delete("/images/{image_id}")
async def delete_image(project_id: str, image_id: str):
    if not await dataset_service.delete_image(project_id, image_id):
        raise HTTPException(status_code=404, detail="Image not found")
    return {"ok": True}


@router.put("/images/{image_id}/category")
async def update_image_category(project_id: str, image_id: str, data: UpdateImageCategory):
    if not await dataset_service.update_image_category(project_id, image_id, data.category):
        raise HTTPException(status_code=404, detail="Image not found")
    return {"ok": True}


@router.put("/images/{image_id}/split")
async def update_image_split(project_id: str, image_id: str, data: UpdateImageSplit):
    if not await dataset_service.update_image_split(project_id, image_id, data.split):
        raise HTTPException(status_code=404, detail="Image not found")
    return {"ok": True}


@router.post("/split")
async def apply_split(project_id: str, data: SplitRequest):
    count = await dataset_service.apply_split(project_id, data.train_ratio, data.seed)
    return {"ok": True, "total_images": count}

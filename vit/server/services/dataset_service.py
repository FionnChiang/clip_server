import io
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from sqlalchemy import select, func, delete, update

from ..s3_client import s3_client
from ..database import get_session, Project, DatasetImage
from ..document_converter import (
    SUPPORTED_DOC_EXTENSIONS, document_to_images,
)
from ..schemas.dataset import (
    CategoryOut, DatasetImageOut, PaginatedImages,
    SplitRequest, UpdateImageCategory, UpdateImageSplit,
)

logger = logging.getLogger(__name__)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


async def list_categories(project_id: str) -> list[CategoryOut]:
    async with get_session() as session:
        rows = await session.execute(
            select(
                DatasetImage.category,
                func.count(DatasetImage.id).label("total"),
                func.sum(func.if_(DatasetImage.split == "train", 1, 0)).label("train"),
                func.sum(func.if_(DatasetImage.split == "val", 1, 0)).label("val"),
            )
            .where(DatasetImage.project_id == project_id)
            .group_by(DatasetImage.category)
            .order_by(DatasetImage.category)
        )
        cats = []
        for cat, total, train, val in rows:
            cats.append(CategoryOut(
                name=cat,
                image_count=total or 0,
                train_count=train or 0,
                val_count=val or 0,
            ))
        return cats


async def add_category(project_id: str, name: str) -> None:
    pass


async def remove_category(project_id: str, name: str) -> int:
    async with get_session() as session:
        images_result = await session.execute(
            select(DatasetImage).where(
                DatasetImage.project_id == project_id,
                DatasetImage.category == name,
            )
        )
        images = images_result.scalars().all()
        for img in images:
            key = _key_from_url(img.s3_url)
            if key:
                try:
                    s3_client.delete(key)
                except Exception as e:
                    logger.warning(f"Failed to delete S3 object {key}: {e}")
            await session.delete(img)

        project_result = await session.execute(select(Project).where(Project.id == project_id))
        project = project_result.scalar()
        if project:
            cats = list(project.categories or [])
            if name in cats:
                cats.remove(name)
                project.categories = cats
        return len(images)


async def upload_images(
    project_id: str,
    category: str,
    files: list[tuple[str, bytes, str]],
) -> list[DatasetImageOut]:
    async with get_session() as session:
        project_result = await session.execute(select(Project).where(Project.id == project_id))
        project = project_result.scalar()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        cats = list(project.categories or [])
        if category not in cats:
            cats.append(category)
            project.categories = cats

        results = []
        for filename, data, content_type in files:
            for page_filename, page_data, page_ctype in _expand_file(filename, data, content_type):
                results.append(
                    await _save_image(session, project_id, category, page_filename, page_data, page_ctype)
                )

        return results


def _expand_file(filename: str, data: bytes, content_type: str) -> list[tuple[str, bytes, str]]:
    """将上传文件展开为 (文件名, 字节, content_type) 图片列表。

    - 普通图片：原样返回。
    - PDF/OFD：按页渲染为 PNG，每页一张，文件名带页码后缀。
    - 其他类型：返回空列表（跳过）。

    Raises:
        ValueError: 文档解析失败（加密/损坏/无有效页面）。
    """
    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return [(filename, data, content_type)]
    if ext in SUPPORTED_DOC_EXTENSIONS:
        stem = Path(filename).stem
        return [
            (f"{stem}_p{page_no:03d}.png", png_bytes, "image/png")
            for page_no, png_bytes in document_to_images(filename, data)
        ]
    return []


async def _save_image(
    session,
    project_id: str,
    category: str,
    filename: str,
    data: bytes,
    content_type: str,
) -> DatasetImageOut:
    """上传单张图片到 S3 并写入 DatasetImage 记录。"""
    try:
        pil_img = Image.open(io.BytesIO(data))
        width, height = pil_img.size
    except Exception:
        width, height = 0, 0

    file_size = len(data)
    s3_key = f"projects/{project_id}/{category}/{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{filename}"
    s3_url = s3_client.upload_bytes(s3_key, data, content_type or "image/jpeg")

    img_record = DatasetImage(
        project_id=project_id,
        category=category,
        s3_url=s3_url,
        original_filename=filename,
        file_size=file_size,
        width=width,
        height=height,
    )
    session.add(img_record)
    await session.flush()

    return DatasetImageOut(
        id=img_record.id,
        category=img_record.category,
        s3_url=img_record.s3_url,
        original_filename=img_record.original_filename,
        file_size=img_record.file_size or 0,
        width=img_record.width or 0,
        height=img_record.height or 0,
        split=img_record.split or "unassigned",
        uploaded_at=img_record.uploaded_at,
    )


async def list_images(
    project_id: str,
    category: Optional[str] = None,
    split: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> PaginatedImages:
    async with get_session() as session:
        q = select(DatasetImage).where(DatasetImage.project_id == project_id)
        count_q = select(func.count(DatasetImage.id)).where(DatasetImage.project_id == project_id)

        if category:
            q = q.where(DatasetImage.category == category)
            count_q = count_q.where(DatasetImage.category == category)
        if split:
            q = q.where(DatasetImage.split == split)
            count_q = count_q.where(DatasetImage.split == split)

        total_result = await session.execute(count_q)
        total = total_result.scalar() or 0

        q = q.order_by(DatasetImage.uploaded_at.desc())
        q = q.offset((page - 1) * page_size).limit(page_size)
        result = await session.execute(q)
        images = result.scalars().all()

        items = [
            DatasetImageOut(
                id=img.id,
                category=img.category,
                s3_url=img.s3_url,
                original_filename=img.original_filename or "",
                file_size=img.file_size or 0,
                width=img.width or 0,
                height=img.height or 0,
                split=img.split or "unassigned",
                uploaded_at=img.uploaded_at,
            )
            for img in images
        ]

        return PaginatedImages(items=items, total=total, page=page, page_size=page_size)


async def delete_image(project_id: str, image_id: str) -> bool:
    async with get_session() as session:
        result = await session.execute(
            select(DatasetImage).where(
                DatasetImage.id == image_id,
                DatasetImage.project_id == project_id,
            )
        )
        img = result.scalar()
        if not img:
            return False
        key = _key_from_url(img.s3_url)
        if key:
            try:
                s3_client.delete(key)
            except Exception as e:
                logger.warning(f"Failed to delete S3 object {key}: {e}")
        await session.delete(img)
        return True


async def update_image_category(project_id: str, image_id: str, category: str) -> bool:
    async with get_session() as session:
        result = await session.execute(
            select(DatasetImage).where(
                DatasetImage.id == image_id,
                DatasetImage.project_id == project_id,
            )
        )
        img = result.scalar()
        if not img:
            return False
        img.category = category
        return True


async def update_image_split(project_id: str, image_id: str, split: str) -> bool:
    async with get_session() as session:
        result = await session.execute(
            select(DatasetImage).where(
                DatasetImage.id == image_id,
                DatasetImage.project_id == project_id,
            )
        )
        img = result.scalar()
        if not img:
            return False
        img.split = split
        return True


async def apply_split(project_id: str, train_ratio: float = 0.8, seed: int = 42) -> int:
    async with get_session() as session:
        result = await session.execute(
            select(DatasetImage).where(DatasetImage.project_id == project_id)
        )
        images = result.scalars().all()

        random.seed(seed)
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        for i, img in enumerate(images):
            img.split = "train" if i < split_idx else "val"

        return len(images)


def _key_from_url(url: str) -> Optional[str]:
    if url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        if len(parts) == 2:
            return parts[1]
    return None

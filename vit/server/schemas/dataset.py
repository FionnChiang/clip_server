from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class CategoryOut(BaseModel):
    name: str
    image_count: int
    train_count: int = 0
    val_count: int = 0


class CategoryCreate(BaseModel):
    name: str


class DatasetImageOut(BaseModel):
    id: str
    category: str
    s3_url: str
    original_filename: str
    file_size: int
    width: int
    height: int
    split: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class PaginatedImages(BaseModel):
    items: list[DatasetImageOut]
    total: int
    page: int
    page_size: int


class SplitRequest(BaseModel):
    train_ratio: float = 0.8
    seed: int = 42


class UpdateImageCategory(BaseModel):
    category: str


class UpdateImageSplit(BaseModel):
    split: str

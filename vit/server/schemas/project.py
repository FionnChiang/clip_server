from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    model_path: str = "../models"


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_path: Optional[str] = None
    categories: Optional[list[str]] = None
    config: Optional[dict] = None
    status: Optional[str] = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    model_path: str
    categories: list[str]
    config: dict
    status: str
    created_at: datetime
    updated_at: datetime
    image_count: int = 0
    model_count: int = 0
    latest_job_status: Optional[str] = None

    class Config:
        from_attributes = True

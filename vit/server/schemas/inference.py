from typing import Optional
from pydantic import BaseModel


class PredictionResult(BaseModel):
    category: str
    index: int
    confidence: float
    probabilities: dict[str, float]
    rejected: bool = False
    reason: Optional[str] = None
    original_category: Optional[str] = None
    original_index: Optional[int] = None


class BatchPredictionResult(BaseModel):
    results: list[PredictionResult]


class PagePredictionResult(BaseModel):
    """文档单页预测结果。"""

    page: int
    category: str
    index: int
    confidence: float
    probabilities: dict[str, float]
    rejected: bool = False
    reason: Optional[str] = None
    original_category: Optional[str] = None
    original_index: Optional[int] = None


class DocumentPredictionResult(BaseModel):
    """文档（PDF/OFD）逐页预测结果。"""

    filename: str
    page_count: int
    results: list[PagePredictionResult]


class PageTopKResult(BaseModel):
    """文档单页 Top-K 预测结果。"""

    page: int
    top_k: list[PredictionResult]


class DocumentTopKResult(BaseModel):
    """文档（PDF/OFD）逐页 Top-K 预测结果。"""

    filename: str
    page_count: int
    results: list[PageTopKResult]

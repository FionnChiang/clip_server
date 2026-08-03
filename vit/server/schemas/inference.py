from typing import Optional
from pydantic import BaseModel


class PredictionResult(BaseModel):
    category: str
    index: int
    confidence: float
    probabilities: dict[str, float]


class BatchPredictionResult(BaseModel):
    results: list[PredictionResult]

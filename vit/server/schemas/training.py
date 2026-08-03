from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    batch_size: int = 32
    epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 0.0001
    lr_scheduler: str = "cosine"
    early_stop_patience: int = 10
    class_balance: str = "weighted_loss"
    num_workers: int = 4
    save_best_only: bool = True
    mixed_precision: bool = False
    log_interval: int = 10


class ModelConfig(BaseModel):
    path: str = "../models"
    freeze_encoder: bool = True
    dropout: float = 0.1
    projection_dim: Optional[int] = None
    pool: str = "cls"


class StartTrainingRequest(BaseModel):
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    train_ratio: float = 0.8
    seed: int = 42


class TrainingJobOut(BaseModel):
    id: str
    project_id: str
    status: str
    current_epoch: int
    total_epochs: int
    best_val_acc: float
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


class TrainingMetricOut(BaseModel):
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float

    class Config:
        from_attributes = True


class WSTrainingMessage(BaseModel):
    type: str
    job_id: str
    data: dict

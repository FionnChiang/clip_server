import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

from sqlalchemy import (
    Column, String, Text, Integer, Float, BigInteger,
    DateTime, ForeignKey, JSON, create_engine,
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import server_config

Base = declarative_base()


def _uuid():
    return uuid.uuid4().hex


class Project(Base):
    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    model_path = Column(String(500), default="../models")
    categories = Column(JSON, default=list)
    config = Column(JSON, default=dict)
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    images = relationship("DatasetImage", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("TrainingJob", back_populates="project", cascade="all, delete-orphan")
    models = relationship("Model", back_populates="project", cascade="all, delete-orphan")


class DatasetImage(Base):
    __tablename__ = "dataset_images"

    id = Column(String(36), primary_key=True, default=_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    category = Column(String(255), nullable=False)
    s3_url = Column(String(1000), nullable=False)
    original_filename = Column(String(500))
    file_size = Column(BigInteger, default=0)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    split = Column(String(20), default="unassigned")
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="images")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(String(36), primary_key=True, default=_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), default="pending")
    config = Column(JSON, default=dict)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    best_val_acc = Column(Float, default=0.0)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)

    project = relationship("Project", back_populates="jobs")
    metrics = relationship("TrainingMetric", back_populates="job", cascade="all, delete-orphan")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float)
    train_acc = Column(Float)
    val_loss = Column(Float)
    val_acc = Column(Float)
    lr = Column(Float)

    job = relationship("TrainingJob", back_populates="metrics")


class Model(Base):
    __tablename__ = "models"

    id = Column(String(36), primary_key=True, default=_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    job_id = Column(String(36), ForeignKey("training_jobs.id", ondelete="SET NULL"))
    name = Column(String(255))
    checkpoint_path = Column(String(500))
    val_acc = Column(Float, default=0.0)
    categories = Column(JSON, default=list)
    config = Column(JSON, default=dict)
    status = Column(String(20), default="available")
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="models")
    job = relationship("TrainingJob", backref="model")


class InferenceService(Base):
    __tablename__ = "inference_services"

    id = Column(String(36), primary_key=True, default=_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(String(36), ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    port = Column(Integer, default=0)
    status = Column(String(20), default="stopped")
    process_id = Column(Integer)
    started_at = Column(DateTime)


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key_name = Column(String(100), primary_key=True)
    value = Column(Text, default="")
    description = Column(String(500), default="")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


_engine = None
_async_session_factory = None
_sync_session_factory = None


def _build_mysql_url() -> str:
    return server_config.mysql_url


async def init_db():
    global _engine, _async_session_factory, _sync_session_factory
    url = _build_mysql_url()

    sync_url = url.replace("mysql+aiomysql://", "mysql+pymysql://")
    sync_engine = create_engine(sync_url, pool_pre_ping=True)
    _sync_session_factory = sessionmaker(bind=sync_engine)

    Base.metadata.create_all(sync_engine)

    _engine = create_async_engine(
        url,
        pool_size=server_config.mysql_pool_size,
        pool_recycle=server_config.mysql_pool_recycle,
        pool_pre_ping=True,
    )
    _async_session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def close_db():
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


@asynccontextmanager
async def get_session():
    if _async_session_factory is None:
        await init_db()
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_sync_session():
    if _sync_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _sync_session_factory()

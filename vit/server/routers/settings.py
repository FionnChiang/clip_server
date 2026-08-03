import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..s3_client import s3_client
from ..database import get_session, init_db
from ..config import server_config
import sqlalchemy as sa

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/settings", tags=["settings"])


class MySQLConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str


class S3Config(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    region: str
    use_ssl: bool


class SystemSettingsOut(BaseModel):
    mysql: MySQLConfig
    s3: S3Config


@router.get("")
async def get_settings():
    return {
        "mysql": {
            "host": server_config.mysql_host,
            "port": server_config.mysql_port,
            "user": server_config.mysql_user,
            "password": "********" if server_config.mysql_password else "",
            "database": server_config.mysql_database,
        },
        "s3": {
            "endpoint": server_config.s3_endpoint,
            "access_key": server_config.s3_access_key,
            "secret_key": "********" if server_config.s3_secret_key else "",
            "bucket": server_config.s3_bucket,
            "region": server_config.s3_region,
            "use_ssl": server_config.s3_use_ssl,
        },
    }


@router.put("/mysql")
async def update_mysql(config: MySQLConfig):
    server_config.update_mysql(
        config.host, config.port, config.user,
        config.password, config.database,
    )
    return {"ok": True}


@router.post("/test-mysql")
async def test_mysql(config: MySQLConfig):
    url = f"mysql+aiomysql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}"
    try:
        engine = sa.create_engine(url.replace("mysql+aiomysql://", "mysql+pymysql://"))
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        return {"ok": True, "message": "MySQL connection successful"}
    except Exception as e:
        return {"ok": False, "message": str(e)}


@router.put("/s3")
async def update_s3(config: S3Config):
    server_config.update_s3(
        config.endpoint, config.access_key, config.secret_key,
        config.bucket, config.region, config.use_ssl,
    )
    return {"ok": True}


@router.post("/test-s3")
async def test_s3():
    result = s3_client.test_connection()
    return result

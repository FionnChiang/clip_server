import os
from pathlib import Path
import yaml


class ServerConfig:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.environ.get(
                "SERVER_CONFIG",
                str(Path(__file__).resolve().parent.parent / "configs" / "server_config.yaml"),
            )
        self.config_path = config_path
        self._data = {}
        self.reload()

    def reload(self):
        path = Path(self.config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}

    def save(self):
        path = Path(self.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, allow_unicode=True, default_flow_style=False)

    # --- Server ---
    @property
    def host(self) -> str:
        return self._data.get("server", {}).get("host", "0.0.0.0")

    @property
    def port(self) -> int:
        return self._data.get("server", {}).get("port", 8000)

    @property
    def frontend_dist(self) -> str:
        return self._data.get("server", {}).get("frontend_dist", "../frontend/dist")

    # --- MySQL ---
    @property
    def mysql_host(self) -> str:
        return self._data.get("mysql", {}).get("host", "127.0.0.1")

    @property
    def mysql_port(self) -> int:
        return self._data.get("mysql", {}).get("port", 3306)

    @property
    def mysql_user(self) -> str:
        return self._data.get("mysql", {}).get("user", "root")

    @property
    def mysql_password(self) -> str:
        return self._data.get("mysql", {}).get("password", "")

    @property
    def mysql_database(self) -> str:
        return self._data.get("mysql", {}).get("database", "layout_classifier")

    @property
    def mysql_pool_size(self) -> int:
        return self._data.get("mysql", {}).get("pool_size", 10)

    @property
    def mysql_pool_recycle(self) -> int:
        return self._data.get("mysql", {}).get("pool_recycle", 3600)

    @property
    def mysql_url(self) -> str:
        pwd = self.mysql_password
        pwd_part = f":{pwd}" if pwd else ""
        creds = f"{self.mysql_user}{pwd_part}@" if self.mysql_user else ""
        return f"mysql+aiomysql://{creds}{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"

    def update_mysql(self, host, port, user, password, database):
        self._data.setdefault("mysql", {})
        self._data["mysql"].update({
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        })
        self.save()

    # --- S3 ---
    @property
    def s3_endpoint(self) -> str:
        return self._data.get("s3", {}).get("endpoint", "")

    @property
    def s3_access_key(self) -> str:
        return self._data.get("s3", {}).get("access_key", "")

    @property
    def s3_secret_key(self) -> str:
        return self._data.get("s3", {}).get("secret_key", "")

    @property
    def s3_bucket(self) -> str:
        return self._data.get("s3", {}).get("bucket", "layout-classifier")

    @property
    def s3_region(self) -> str:
        return self._data.get("s3", {}).get("region", "us-east-1")

    @property
    def s3_use_ssl(self) -> bool:
        return self._data.get("s3", {}).get("use_ssl", True)

    def update_s3(self, endpoint, access_key, secret_key, bucket, region, use_ssl):
        self._data.setdefault("s3", {})
        self._data["s3"].update({
            "endpoint": endpoint,
            "access_key": access_key,
            "secret_key": secret_key,
            "bucket": bucket,
            "region": region,
            "use_ssl": use_ssl,
        })
        self.save()

    # --- Services ---
    @property
    def training_url(self) -> str:
        return os.environ.get("TRAINING_SERVICE_URL") or self._data.get("services", {}).get("training_url", "http://127.0.0.1:8001")

    @property
    def inference_url(self) -> str:
        return os.environ.get("INFERENCE_SERVICE_URL") or self._data.get("services", {}).get("inference_url", "http://127.0.0.1:8002")

    # --- Training ---
    @property
    def temp_dir(self) -> str:
        return self._data.get("training", {}).get("temp_dir", "../tmp")

    @property
    def raw_config(self) -> dict:
        return self._data


server_config = ServerConfig()

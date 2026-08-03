import io
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from .config import server_config


class S3Client:
    def __init__(self):
        self._client = None

    def _build_client(self):
        cfg = {
            "service_name": "s3",
            "aws_access_key_id": server_config.s3_access_key,
            "aws_secret_access_key": server_config.s3_secret_key,
            "region_name": server_config.s3_region,
            "config": Config(signature_version="s3v4"),
        }
        if server_config.s3_endpoint:
            cfg["endpoint_url"] = server_config.s3_endpoint
            cfg["use_ssl"] = server_config.s3_use_ssl
        return boto3.client(**cfg)

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    @property
    def bucket(self) -> str:
        return server_config.s3_bucket

    def ensure_bucket(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def upload_bytes(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        self.ensure_bucket()
        self.client.upload_fileobj(
            io.BytesIO(data),
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        return self._object_url(key)

    def upload_file(self, key: str, file_path: str, content_type: str = "image/jpeg") -> str:
        self.ensure_bucket()
        self.client.upload_file(
            file_path,
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        return self._object_url(key)

    def download_bytes(self, key: str) -> bytes:
        buf = io.BytesIO()
        self.client.download_fileobj(self.bucket, key, buf)
        return buf.getvalue()

    def download_file(self, key: str, file_path: str):
        self.client.download_file(self.bucket, key, file_path)

    def delete(self, key: str):
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def get_presigned_url(self, key: str, expires: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )

    def _object_url(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def test_connection(self) -> dict:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return {"ok": True, "message": f"Connected to bucket '{self.bucket}' successfully"}
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "404":
                return {"ok": True, "message": f"Connected to S3 successfully. Bucket '{self.bucket}' does not exist yet (will be auto-created)."}
            return {"ok": False, "message": str(e)}
        except Exception as e:
            return {"ok": False, "message": str(e)}


s3_client = S3Client()

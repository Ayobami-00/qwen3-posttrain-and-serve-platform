import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from qps.specs import Contract


class Settings(Contract):
    database_url: str = "sqlite+pysqlite:///:memory:"
    bucket: str = "qps-development"
    object_prefix: str = "test"
    s3_endpoint: str | None = None
    aws_region: str = "eu-west-2"
    public_url: str = "http://127.0.0.1:8080"
    auth_mode: Literal["cognito", "development"] = "cognito"
    dev_token: str = ""
    internal_token: str = Field(min_length=32)
    worker_signing_key: str = Field(min_length=32)
    session_key: str = ""
    cognito_issuer: str = ""
    cognito_client_id: str = ""
    cognito_domain: str = ""
    namespace: str = "qps"
    cpu_image: str = ""
    gpu_image: str = ""
    dispatch_enabled: bool = True
    forge_enabled: bool = False
    forge_entity: str = "enfinity7b"
    forge_project: str = "qps-platform"
    forge_base_url: str = "https://api.forge.coreweave.com"
    forge_report_url: str | None = None
    max_gpu_allocations: int = Field(default=2, ge=1, le=2)
    runpod_budget_usd: float = Field(default=3, gt=0, le=3)
    max_gpu_hourly_usd: float = Field(default=1, gt=0, le=1)
    lease_seconds: int = Field(default=90, ge=30, le=180)
    allocation_seconds: int = Field(default=2700, ge=60, le=2700)
    upload_limit_bytes: int = 40 * 1024**3

    @model_validator(mode="after")
    def production(self):
        if self.auth_mode == "development":
            if not self.dev_token or not self.public_url.startswith(
                ("http://127.0.0.1:", "http://localhost:")
            ):
                raise ValueError("Development auth requires a token and a loopback public URL")
        else:
            if not self.database_url.startswith("postgresql"):
                raise ValueError("Hosted production requires PostgreSQL")
            if not self.public_url.startswith("https://"):
                raise ValueError("Production public URL must use HTTPS")
            if not all(
                (self.cognito_issuer, self.cognito_client_id, self.cognito_domain, self.session_key)
            ):
                raise ValueError("Cognito issuer, client, domain and session key are required")
        for image in (self.cpu_image, self.gpu_image):
            if image and "@sha256:" not in image and self.auth_mode != "development":
                raise ValueError("Production worker images must be pinned by digest")
        return self

    @classmethod
    def from_env(cls):
        values = {}
        for key, field in cls.model_fields.items():
            name = "QPS_" + key.upper()
            if name + "_FILE" in os.environ:
                values[key] = Path(os.environ[name + "_FILE"]).read_text().strip()
            elif name in os.environ:
                values[key] = os.environ[name]
        return cls(**values)

"""Regenerate checked-in contracts; run from the repository root."""

import json
from pathlib import Path

from qps.contracts import (
    BackendCapability,
    DeploymentSubmission,
    HostedSubmission,
    WorkerBatch,
)
from qps.specs import RunSpec

root = Path(__file__).resolve().parents[1]
for name, model in {
    "run": RunSpec,
    "hosted-run": HostedSubmission,
    "deployment": DeploymentSubmission,
    "worker-events": WorkerBatch,
    "backend": BackendCapability,
}.items():
    (root / "contracts" / f"{name}.schema.json").write_text(
        json.dumps(model.model_json_schema(by_alias=True), indent=2) + "\n"
    )
(root / "contracts/fixtures/python-backend.json").write_text(
    json.dumps(BackendCapability().model_dump(by_alias=True), indent=2) + "\n"
)

# Exporting OpenAPI must not need cloud credentials, model imports or live services.
from qps.hosted.api import create_app
from qps.hosted.database import Repository
from qps.hosted.settings import Settings

settings = Settings(
    auth_mode="development",
    dev_token="schema-only",
    internal_token="schema-only-" * 4,
    worker_signing_key="schema-only-" * 4,
    dispatch_enabled=False,
)
api = create_app(settings, Repository(settings, create=True), objects=object())
schema = api.openapi()
schema["components"]["securitySchemes"] = {
    name: {"type": "http", "scheme": "bearer"}
    for name in ("OwnerBearer", "ControllerBearer", "WorkerBearer")
}
for path, operations in schema["paths"].items():
    for method, operation in operations.items():
        if not isinstance(operation, dict):
            continue
        if path.startswith("/api/internal/"):
            operation["security"] = [{"ControllerBearer": []}]
        elif path.startswith("/api/worker/"):
            operation["security"] = [{"WorkerBearer": []}]
        elif method in ("post", "put", "delete", "patch") or path.startswith("/v1/"):
            operation["security"] = [{"OwnerBearer": []}]
(root / "contracts/openapi.json").write_text(json.dumps(schema, indent=2) + "\n")

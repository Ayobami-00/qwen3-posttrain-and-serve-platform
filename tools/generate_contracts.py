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

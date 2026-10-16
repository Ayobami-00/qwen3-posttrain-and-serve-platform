"""Hosted contracts; deliberately independent of cloud SDKs and model libraries."""

from datetime import datetime
from typing import Literal, Protocol

from pydantic import Field, model_validator

from qps.specs import Contract, Metadata, RunSpec
from qps.util import fingerprint

API_VERSION = "qps.dev/v1alpha1"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
ARTIFACT_KINDS = {
    "base_model",
    "dataset",
    "training_checkpoint",
    "lora_adapter",
    "merged_model",
    "evaluation_report",
    "comparison_report",
    "training_report",
}


class HostedSubmission(Contract):
    run: RunSpec
    compute_profile: Literal["runpod-qwen3-small", "runpod-qwen3-fsdp2", "cpu-fixture"] = (
        "runpod-qwen3-small"
    )
    dataset_version_id: str = "gsm8k-training-v1"
    idempotency_key: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def supported(self):
        workload = self.run.spec
        cpu = self.compute_profile == "cpu-fixture"
        if cpu != (workload.execution.profile == "local-cpu-smoke"):
            raise ValueError("Compute profile and workload execution profile disagree")
        if (self.compute_profile == "runpod-qwen3-fsdp2") != (
            workload.execution.parallelism == "fsdp2"
        ):
            raise ValueError("FSDP2 requires the two-GPU compute profile")
        if not cpu and (
            workload.model.source != "huggingface" or workload.dataset.source != "huggingface"
        ):
            raise ValueError("Hosted GPU runs require the supported Hub model and dataset")
        if workload.execution.max_wall_time_seconds > 2700:
            raise ValueError("Hosted runs have a maximum 2700-second active runtime")
        if workload.training.max_sequence_length > 2048:
            raise ValueError("Hosted training supports at most 2048 sequence tokens")
        return self


class DeploymentSubmission(Contract):
    name: str
    artifact_id: str | None = Field(default=None, pattern=r"^art-[a-f0-9]{32}$")
    model_version_id: str | None = Field(default=None, pattern=r"^model-[a-f0-9]{32}$")
    backend: Literal["python-transformers"] = "python-transformers"
    compute_profile: Literal["runpod-qwen3-small", "cpu-fixture"] = "runpod-qwen3-small"
    max_context_tokens: int = Field(default=2048, ge=16, le=2048)
    max_output_tokens: int = Field(default=512, ge=1, le=512)
    max_runtime_seconds: int = Field(default=900, ge=60, le=2700)
    idempotency_key: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def valid_name(self):
        Metadata(name=self.name)
        if not self.artifact_id and not self.model_version_id:
            raise ValueError("Provide modelVersionId or artifactId")
        return self


class WorkerEvent(Contract):
    sequence: int = Field(ge=1)
    time: datetime
    kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    payload: dict


class WorkerBatch(Contract):
    events: list[WorkerEvent] = Field(max_length=100)


class Heartbeat(Contract):
    phase: str = Field(max_length=64)
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    telemetry: dict = Field(default_factory=dict)


class PhaseOutput(Contract):
    phase: Literal["prepare", "baseline", "training", "export", "evaluation"]
    output: dict


class BackendCapability(Contract):
    id: Literal["python-transformers"] = "python-transformers"
    artifact_kind: Literal["merged_model"] = "merged_model"
    manifest_version: Literal[1] = 1
    architectures: list[str] = Field(default_factory=lambda: ["qwen3"])
    dtypes: list[str] = Field(default_factory=lambda: ["fp32", "bf16"])
    gpu_counts: list[int] = Field(default_factory=lambda: [0, 1])
    max_context_tokens: int = 2048
    max_output_tokens: int = 512
    streaming: bool = True
    max_active_generations: int = 1


class EventSink(Protocol):
    def emit(self, run_id: str, attempt: int, kind: str, payload: dict): ...


class PhaseOutputs(Protocol):
    def output(self, run_id: str, phase: str): ...


class ArtifactPublisher(Protocol):
    def publish(self, run_id: str, kind: str, source, metadata: dict) -> str: ...


def portable_recipe(spec: RunSpec) -> dict:
    """Placement and disposable paths cannot change optimizer compatibility."""
    value = spec.model_dump(mode="json", by_alias=True)
    value["spec"]["artifacts"].pop("root", None)
    # Preserve existing version-2 single-device recipe identities.
    if value["spec"]["execution"].get("parallelism") == "single":
        value["spec"]["execution"].pop("parallelism", None)
    for name in ("model", "dataset"):
        if value["spec"][name]["source"] == "local":
            value["spec"][name]["id"] = "local-snapshot"
    return value


def recipe_digest(spec: RunSpec) -> str:
    return fingerprint(portable_recipe(spec))


class ModelVersion(Contract):
    """Published, independently owned serving identity. API uses database field names."""

    id: str = Field(pattern=r"^model-[a-f0-9]{32}$")
    name: str
    source_run_id: str
    adapter_id: str
    artifact_id: str
    snapshot: dict
    status: Literal["ready", "deleting"]
    created: float


class DeletionOperation(Contract):
    id: str = Field(pattern=r"^op-[a-f0-9]{32}$")
    resource_id: str
    kind: Literal["experiment", "model", "deployment"]
    state: Literal["pending", "running", "retrying", "completed"]
    phase: str
    plan: dict
    error: str | None = None
    created: float
    updated: float
    locked_until: float

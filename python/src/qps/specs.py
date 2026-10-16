"""Lightweight public contracts. Importing this module never imports the ML stack."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", alias_generator=to_camel, populate_by_name=True)


class Metadata(Contract):
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")


class ModelSpec(Contract):
    id: str
    source: Literal["huggingface", "local"] = "huggingface"
    revision: str = "main"
    prompt_profile: Literal["qwen3-chat-nonthinking-v1"] = "qwen3-chat-nonthinking-v1"


class DatasetSpec(Contract):
    id: str
    source: Literal["huggingface", "local"] = "huggingface"
    revision: str = "main"
    configuration: str = "main"
    source_split: Literal["train"] = "train"
    normalizer: Literal["gsm8k-v1"] = "gsm8k-v1"
    split_seed: int = 42
    train_examples: int = Field(default=1024, ge=1)
    validation_examples: int = Field(default=128, ge=1)


class LoRASpec(Contract):
    rank: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[Literal["q_proj", "v_proj"]] = Field(
        default_factory=lambda: ["q_proj", "v_proj"], min_length=1
    )


class TrainingSpec(Contract):
    method: Literal["sft"] = "sft"
    strategy: Literal["lora"] = "lora"
    dtype: Literal["bf16", "fp32"] = "bf16"
    seed: int = 42
    max_sequence_length: int = Field(default=1024, ge=16)
    max_steps: int = Field(default=100, ge=1)
    micro_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=0.0001, gt=0, allow_inf_nan=False)
    packing: Literal[False] = False
    checkpoint_every_steps: int = Field(default=25, ge=1)
    lora: LoRASpec = Field(default_factory=LoRASpec)


class EvaluationSpec(Contract):
    suite: Literal["gsm8k-smoke-validation-v1"] = "gsm8k-smoke-validation-v1"
    generation_profile: Literal["nonthinking-greedy-512-v1"] = "nonthinking-greedy-512-v1"
    compare_with_parent: Literal[True] = True
    verify_merged_export: Literal[True] = True
    max_new_tokens: int = Field(default=512, ge=1, le=512)


class ExecutionSpec(Contract):
    profile: Literal["local-cpu-smoke", "local-cuda"] = "local-cuda"
    parallelism: Literal["single", "fsdp2"] = "single"
    gpu_count: Literal[0, 1, 2] = 1
    max_wall_time_seconds: int = Field(default=7200, ge=1)
    threads: int = Field(default=2, ge=1, le=64)


class ArtifactSpec(Contract):
    root: Path = Path(".qps/artifacts")
    outputs: list[Literal["lora_adapter", "merged_model"]] = Field(
        default_factory=lambda: ["lora_adapter", "merged_model"]
    )

    @model_validator(mode="after")
    def require_outputs(self):
        if sorted(self.outputs) != ["lora_adapter", "merged_model"]:
            raise ValueError("v0.1 requires exactly lora_adapter and merged_model outputs")
        return self


class TrackingSpec(Contract):
    local: Literal[True] = True
    wandb: Literal[False] = False


class WorkloadSpec(Contract):
    model: ModelSpec
    dataset: DatasetSpec
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    artifacts: ArtifactSpec = Field(default_factory=ArtifactSpec)
    tracking: TrackingSpec = Field(default_factory=TrackingSpec)

    @model_validator(mode="after")
    def validate_profile(self):
        cpu = self.execution.profile == "local-cpu-smoke"
        if self.training.dtype != ("fp32" if cpu else "bf16"):
            raise ValueError("CPU smoke requires fp32; local-cuda requires bf16")
        expected = 0 if cpu else 2 if self.execution.parallelism == "fsdp2" else 1
        if self.execution.gpu_count != expected or (cpu and self.execution.parallelism != "single"):
            raise ValueError(
                "CPU requires single/gpuCount: 0; CUDA single requires 1 GPU; fsdp2 requires 2 GPUs"
            )
        if cpu and (self.model.source != "local" or self.dataset.source != "local"):
            raise ValueError("CPU smoke uses local fixture snapshots only")
        if self.model.source == "huggingface" and self.model.id != "Qwen/Qwen3-0.6B":
            raise ValueError("The initial Hub support matrix contains only Qwen/Qwen3-0.6B")
        if self.dataset.source == "huggingface" and self.dataset.id != "openai/gsm8k":
            raise ValueError("The initial Hub dataset support matrix contains only openai/gsm8k")
        return self


class RunSpec(Contract):
    api_version: Literal["qps.dev/v1alpha1"] = "qps.dev/v1alpha1"
    kind: Literal["Run"] = "Run"
    metadata: Metadata
    spec: WorkloadSpec


def load_spec(path: Path) -> RunSpec:
    return RunSpec.model_validate(yaml.safe_load(path.read_text()))


def absolute_spec(spec: RunSpec, cwd: Path | None = None) -> RunSpec:
    result = spec.model_copy(deep=True)
    cwd = cwd or Path.cwd()
    for source in (result.spec.model, result.spec.dataset):
        if source.source == "local":
            source.id = str((cwd / source.id).resolve())
    result.spec.artifacts.root = (cwd / result.spec.artifacts.root).resolve()
    return result

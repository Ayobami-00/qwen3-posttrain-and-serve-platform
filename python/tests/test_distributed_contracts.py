import json

import pytest
from pydantic import ValidationError

from qps.contracts import HostedSubmission, recipe_digest
from qps.distributed import validate_layout
from qps.specs import load_spec


def cuda_spec():
    from pathlib import Path

    spec = load_spec(Path(__file__).resolve().parents[2] / "examples/local/gsm8k-lora.yaml")
    spec.spec.execution.max_wall_time_seconds = 2700
    return spec


def test_two_rank_profile_and_legacy_recipe_identity():
    spec = cuda_spec()
    digest = recipe_digest(spec)
    assert spec.spec.execution.parallelism == "single"
    spec.spec.execution.parallelism = "fsdp2"
    with pytest.raises(ValidationError, match="2 GPUs"):
        type(spec).model_validate(spec.model_dump())
    spec.spec.execution.gpu_count = 2
    spec.spec.training.gradient_accumulation_steps = 8
    request = HostedSubmission(
        run=spec, compute_profile="runpod-qwen3-fsdp2", idempotency_key="distributed-0001"
    )
    assert (
        request.run.spec.training.micro_batch_size
        * request.run.spec.training.gradient_accumulation_steps
        * request.run.spec.execution.gpu_count
        == 16
    )
    with pytest.raises(ValidationError, match="two-GPU"):
        HostedSubmission(
            run=spec, compute_profile="runpod-qwen3-small", idempotency_key="distributed-0002"
        )
    spec.spec.execution.parallelism = "single"
    spec.spec.execution.gpu_count = 1
    spec.spec.training.gradient_accumulation_steps = 16
    assert recipe_digest(spec) == digest


def test_distributed_batch_sampler_has_no_overlap_and_repeats_deterministically():
    import torch
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler, RandomSampler

    def membership(rank):
        sampler = RandomSampler(list(range(32)), generator=torch.Generator().manual_seed(42))
        batches = BatchSampler(sampler, batch_size=1, drop_last=True)
        return list(
            BatchSamplerShard(
                batches,
                num_processes=2,
                process_index=rank,
                split_batches=False,
                even_batches=False,
            )
        )

    first, second = membership(0), membership(1)
    assert first == membership(0) and second == membership(1)
    for offset in range(0, len(first), 8):
        batch = [
            i for part in (first[offset : offset + 8] + second[offset : offset + 8]) for i in part
        ]
        assert len(batch) == len(set(batch)) == 16


def test_dcp_layout_requires_complete_rank_and_optimizer_state(tmp_path):
    meta = {"version": 3, "world_size": 2, "format": "qps-fsdp2-dcp-v1", "global_step": 25}
    with pytest.raises(ValueError, match="missing"):
        validate_layout(tmp_path, meta)
    for name in ("scheduler.pt", "trainer_state.json"):
        (tmp_path / name).touch()
    for component in ("pytorch_model_fsdp_0", "optimizer_0"):
        directory = tmp_path / component
        directory.mkdir()
        (directory / ".metadata").touch()
        for rank in range(2):
            (directory / f"__{rank}_0.distcp").touch()
    for rank in range(2):
        (tmp_path / f"rng_state_{rank}.pth").touch()
        (tmp_path / f"rank-{rank}-progress.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": 2,
                    "global_step": 25,
                    "rng_file": f"rng_state_{rank}.pth",
                }
            )
        )
    validate_layout(tmp_path, meta)
    with pytest.raises(ValueError, match="two-rank"):
        validate_layout(tmp_path, {**meta, "world_size": 1})
    (tmp_path / "optimizer_0/__1_0.distcp").unlink()
    with pytest.raises(ValueError, match="shard"):
        validate_layout(tmp_path, meta)

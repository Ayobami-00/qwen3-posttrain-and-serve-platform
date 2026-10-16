"""Two-rank FSDP2 support for the shared TRL trainer.

Accelerate owns fully_shard and optimizer remapping; torchrun owns rank processes.
No distributed process other than rank zero opens the execution journal.
"""

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


def initialize():
    import torch
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "0")) != 2 or torch.cuda.device_count() != 2:
        raise ValueError("FSDP2 requires torchrun with exactly two visible CUDA devices")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("Both FSDP2 devices must support BF16")
    os.environ.update(
        ACCELERATE_USE_FSDP="true",
        FSDP_VERSION="2",
        FSDP_CPU_RAM_EFFICIENT_LOADING="true",
        FSDP_STATE_DICT_TYPE="SHARDED_STATE_DICT",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="Qwen3DecoderLayer",
        FSDP_ACTIVATION_CHECKPOINTING="true",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
    )
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", timeout=timedelta(seconds=120), device_id=torch.device("cuda", rank)
        )


def barrier():
    import torch.distributed as dist

    dist.barrier()


def main_rank():
    return int(os.environ.get("RANK", "0")) == 0


def finite_everywhere(value, description):
    import torch
    import torch.distributed as dist

    local = value.to_local() if hasattr(value, "to_local") else value
    flag = torch.isfinite(local).all().to(device=torch.cuda.current_device(), dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise ValueError("Non-finite distributed " + description)


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def gather_adapter(model):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True),
    )


def validate_prepared(trainer, optimizer, ctx):
    from torch.distributed.tensor import DTensor

    actual = {id(p) for p in trainer.model.parameters() if p.requires_grad}
    selected = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if actual != selected or not actual:
        raise ValueError("FSDP2 optimizer does not reference the prepared trainable parameters")
    parameters = list(trainer.model.named_parameters())
    if not all(isinstance(p, DTensor) for _, p in parameters):
        raise ValueError("FSDP2 did not shard every model parameter")
    if not any(p.to_local().numel() < p.numel() for _, p in parameters):
        raise ValueError("FSDP2 parameter storage is not sharded")
    ctx.emit(
        "distributed_prepared",
        world_size=2,
        global_parameter_elements=sum(p.numel() for _, p in parameters),
        local_parameter_elements=sum(p.to_local().numel() for _, p in parameters),
        effective_batch_size=ctx.spec.training.micro_batch_size
        * ctx.spec.training.gradient_accumulation_steps
        * 2,
    )


def configure_trainer(trainer):
    import torch
    from torch.distributed.fsdp import MixedPrecisionPolicy

    plugin = trainer.accelerator.state.fsdp_plugin
    if plugin is None or plugin.fsdp_version != 2:
        raise ValueError("Installed Trainer/Accelerate did not select FSDP2")
    plugin.mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    plugin.activation_checkpointing = True
    plugin.cpu_ram_efficient_loading = True


def aggregate(counters, previous, started):
    import torch
    import torch.distributed as dist

    values = torch.tensor(
        [
            counters["tokens"] - previous["tokens"],
            counters["supervised"] - previous["supervised"],
            counters["loss_sum"],
        ],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(values)
    diagnostic = {
        "rank": dist.get_rank(),
        "device": torch.cuda.current_device(),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.monotonic() - started,
    }
    ranks = [None] * 2
    dist.all_gather_object(ranks, diagnostic)
    return {
        "tokens_seen": previous["tokens"] + int(values[0]),
        "supervised_tokens_seen": previous["supervised"] + int(values[1]),
        "loss": float(values[2] / values[1]) if values[1] else 0,
        "ranks": ranks,
        "cuda_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in ranks),
        "cuda_memory_aggregate": "maximum per-rank peak; not a simultaneous sum",
    }


def checkpoint_files(directory, step):
    import torch.distributed as dist

    from qps.util import atomic_json

    atomic_json(
        directory / f"rank-{dist.get_rank()}-progress.json",
        {
            "world_size": 2,
            "rank": dist.get_rank(),
            "global_step": step,
            "rng_file": f"rng_state_{dist.get_rank()}.pth",
        },
    )
    barrier()


def validate_layout(directory, metadata):
    if (
        metadata.get("version") != 3
        or metadata.get("world_size") != 2
        or metadata.get("format") != "qps-fsdp2-dcp-v1"
    ):
        raise ValueError("FSDP2 requires a compatible two-rank distributed checkpoint")
    required = [
        "scheduler.pt",
        "trainer_state.json",
        "pytorch_model_fsdp_0/.metadata",
        "optimizer_0/.metadata",
    ]
    for rank in range(2):
        required += [f"rng_state_{rank}.pth", f"rank-{rank}-progress.json"]
    if not all((directory / p).is_file() for p in required):
        raise ValueError("Distributed checkpoint is missing rank, optimizer or scheduler state")
    for component in ("pytorch_model_fsdp_0", "optimizer_0"):
        if len(list((directory / component).glob("*.distcp"))) < 2:
            raise ValueError("Distributed checkpoint is missing a required shard")
    for rank in range(2):
        progress = json.loads((directory / f"rank-{rank}-progress.json").read_text())
        if progress != {
            "world_size": 2,
            "rank": rank,
            "global_step": metadata["global_step"],
            "rng_file": f"rng_state_{rank}.pth",
        }:
            raise ValueError("Distributed rank progress is incompatible")


class SnapshotContext:
    """A rank's immutable input snapshot: no Store or SQLite writer is constructed."""

    def __init__(self, path):
        from qps.specs import RunSpec

        value = json.loads(Path(path).read_text())
        self.id, self.attempt, self.run = value["id"], value["attempt"], value["run"]
        self.spec = RunSpec.model_validate_json(self.run["spec"]).spec
        self.work = Path(value["work"])
        self.inputs = value
        self.store = SimpleNamespace(
            artifact=lambda identity: value["artifacts"][identity]["manifest"]
        )

    def emit(self, *args, **kwargs):
        return None

    def prepared(self):
        return self.inputs["prepared"]

    def artifact(self, identity):
        item = self.inputs["artifacts"][identity]
        return Path(item["directory"]), item["manifest"]

    def records(self):
        return self.inputs["records"]


def write_snapshot(ctx, resume_id):
    from qps.hosted.lifecycle import references
    from qps.util import atomic_json

    identities = references(ctx.prepared())
    if resume_id:
        identities.add(resume_id)
    artifacts = {}
    for identity in identities:
        if not isinstance(identity, str) or not identity.startswith("art-"):
            continue
        directory, manifest = ctx.artifact(identity)
        artifacts[identity] = {"directory": str(directory), "manifest": manifest}
    path = ctx.work / "distributed-input.json"
    atomic_json(
        path,
        {
            "id": ctx.id,
            "attempt": ctx.attempt,
            "run": ctx.run,
            "work": str(ctx.work),
            "prepared": ctx.prepared(),
            "records": ctx.records(),
            "artifacts": artifacts,
        },
    )
    return path


def rank_health(directory, step):
    import torch.distributed as dist

    from qps.util import atomic_json

    atomic_json(
        directory / f"rank-{dist.get_rank()}-health.json",
        {"rank": dist.get_rank(), "time": time.time(), "global_step": step},
    )


def finalize_checkpoint(trainer, ctx, directory, metadata, checkpoint_id):
    from types import SimpleNamespace

    import torch
    from accelerate.utils.fsdp_utils import load_fsdp_model, load_fsdp_optimizer
    from transformers import TrainerState

    trainer.create_optimizer_and_scheduler(num_training_steps=trainer.args.max_steps)
    trainer.model, trainer.optimizer = trainer.accelerator.prepare(trainer.model, trainer.optimizer)
    plugin = trainer.accelerator.state.fsdp_plugin
    load_fsdp_model(plugin, trainer.accelerator, trainer.model, str(directory), adapter_only=True)
    load_fsdp_optimizer(
        plugin, trainer.accelerator, trainer.optimizer, trainer.model, str(directory)
    )
    trainer.lr_scheduler.load_state_dict(
        torch.load(directory / "scheduler.pt", map_location="cpu", weights_only=True)
    )
    trainer.state = TrainerState.load_from_json(str(directory / "trainer_state.json"))
    trainer._load_rng_state(str(directory))
    validate_prepared(trainer, trainer.optimizer, ctx)
    steps = [float(v["step"]) for v in trainer.optimizer.state.values() if "step" in v]
    if (
        trainer.state.global_step != metadata["global_step"]
        or not steps
        or min(steps) < metadata["global_step"]
    ):
        raise ValueError("Completed checkpoint did not restore optimizer progress")
    ctx.emit(
        "training_finalized_from_checkpoint",
        global_step=metadata["global_step"],
        checkpoint=checkpoint_id,
        optimizer_min_step=min(steps),
    )
    return SimpleNamespace(global_step=metadata["global_step"])

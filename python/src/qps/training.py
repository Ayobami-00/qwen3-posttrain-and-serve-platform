"""Real TRL/PEFT SFT with explicit masks, durable checkpoints and finite-value checks."""

import json
import math
import resource
import sys
import time
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from qps.artifacts import file_manifest, verify_directory
from qps.contracts import recipe_digest
from qps.ml import collate, load_model, tokenizer_from
from qps.specs import RunSpec
from qps.stages import Context, environment
from qps.util import atomic_json, fingerprint


def checkpoint_metadata(ctx: Context, tokens_seen: int, supervised: int, step: int):
    return {
        "version": 2,
        "run_id": ctx.id,
        "spec_sha": recipe_digest(RunSpec.model_validate_json(ctx.run["spec"])),
        "parent": ctx.prepared()["parent"],
        "dataset": ctx.prepared()["dataset"],
        "environment": environment(),
        "global_step": step,
        "tokens_seen": tokens_seen,
        "supervised_tokens_seen": supervised,
    }


def validate_checkpoint(ctx: Context, directory: Path):
    metadata = json.loads((directory / "qps-checkpoint.json").read_text())
    actual = checkpoint_metadata(ctx, 0, 0, 0)
    if metadata.get("version") == 1:
        actual["spec_sha"] = fingerprint(json.loads(ctx.run["spec"]))
    for key in ("run_id", "spec_sha", "parent", "dataset", "environment"):
        if metadata[key] != actual[key]:
            raise ValueError(f"Checkpoint {key} is incompatible with this run/environment")
    # The artifact envelope already verifies every byte, including optimizer/RNG files.
    required = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "rng_state.pth",
    }
    if not all((directory / name).is_file() for name in required):
        raise ValueError(
            "Checkpoint is incomplete; optimizer/scheduler/RNG/trainer state is required"
        )
    return metadata


def train(ctx: Context, resume_id: str | None = None):
    set_seed(ctx.spec.training.seed)
    parent_id = ctx.prepared()["parent"]
    directory, _ = ctx.artifact(parent_id)
    tokenizer = tokenizer_from(directory)
    model = load_model(directory, ctx.spec)
    model.config.use_cache = False
    settings = ctx.spec.training
    resume_directory = None
    resume = {"tokens_seen": 0, "supervised_tokens_seen": 0, "global_step": 0}
    if resume_id:
        resume_directory, manifest = ctx.artifact(resume_id)
        if manifest["kind"] != "training_checkpoint":
            raise ValueError("Resume requires a registered training checkpoint")
        resume = validate_checkpoint(ctx, resume_directory)
        if resume["global_step"] > settings.max_steps:
            raise ValueError("Checkpoint exceeds this run's immutable maxSteps")

    counters = {"tokens": resume["tokens_seen"], "supervised": resume["supervised_tokens_seen"]}

    class CheckedTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            if model.training:
                counters["tokens"] += int(inputs["attention_mask"].sum())
                counters["supervised"] += int((inputs["labels"][:, 1:] != -100).sum())
            return (loss, outputs) if return_outputs else loss

    checkpoint_ids = []
    started = time.monotonic()

    class Progress(TrainerCallback):
        last_log = started

        def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
            if resume_directory:
                steps = [
                    float(value["step"]) for value in optimizer.state.values() if "step" in value
                ]
                if (
                    state.global_step != resume["global_step"]
                    or not steps
                    or min(steps) < state.global_step
                ):
                    raise ValueError("Resume failed to restore global step and optimizer progress")
                ctx.emit(
                    "resume_verified",
                    global_step=state.global_step,
                    optimizer_min_step=min(steps),
                    checkpoint=resume_id,
                )

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None or "loss" not in logs:
                return
            current = time.monotonic()
            values = {
                "global_step": state.global_step,
                "loss": float(logs["loss"]),
                "learning_rate": float(logs.get("learning_rate", 0)),
                "grad_norm": float(logs["grad_norm"]) if "grad_norm" in logs else None,
                "elapsed_seconds": current - started,
                "step_seconds": current - self.last_log,
                "tokens_seen": counters["tokens"],
                "supervised_tokens_seen": counters["supervised"],
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated()
                    if ctx.spec.execution.profile == "local-cuda"
                    else None
                ),
            }
            if any(
                isinstance(value, float) and not math.isfinite(value) for value in values.values()
            ):
                raise ValueError("Non-finite training metric")
            ctx.emit("train_step", **values)
            self.last_log = current

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            meta = checkpoint_metadata(
                ctx, counters["tokens"], counters["supervised"], state.global_step
            )
            meta["files"] = file_manifest(checkpoint)
            verify_directory(checkpoint, meta)
            atomic_json(checkpoint / "qps-checkpoint.json", meta)
            validate_checkpoint(ctx, checkpoint)
            artifact_id = ctx.artifacts.publish(
                ctx.id,
                "training_checkpoint",
                checkpoint,
                {
                    "global_step": state.global_step,
                    "parent": parent_id,
                },
            )
            checkpoint_ids.append(artifact_id)
            ctx.emit("checkpoint_saved", global_step=state.global_step, artifact_id=artifact_id)

    args = SFTConfig(
        output_dir=str(ctx.work / "checkpoints"),
        max_steps=settings.max_steps,
        per_device_train_batch_size=settings.micro_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        bf16=settings.dtype == "bf16",
        use_cpu=ctx.spec.execution.profile == "local-cpu-smoke",
        gradient_checkpointing=ctx.spec.execution.profile == "local-cuda",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=settings.max_sequence_length,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        logging_steps=1,
        logging_nan_inf_filter=False,
        report_to="none",
        disable_tqdm=True,
        save_strategy="steps",
        save_steps=settings.checkpoint_every_steps,
        save_only_model=False,
        seed=settings.seed,
        data_seed=settings.seed,
        dataloader_num_workers=0,
        optim="adamw_torch",
        lr_scheduler_type="linear",
    )
    dataset = Dataset.from_list(
        [{key: row[key] for key in ("input_ids", "labels")} for row in ctx.records()["train"]]
    )
    trainer = CheckedTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=partial(collate, pad_token_id=tokenizer.pad_token_id),
        peft_config=LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=settings.lora.rank,
            lora_alpha=settings.lora.alpha,
            lora_dropout=settings.lora.dropout,
            target_modules=settings.lora.target_modules,
            bias="none",
        ),
        callbacks=[Progress()],
    )
    initial = {}

    def check_gradient(gradient):
        if not torch.isfinite(gradient).all():
            raise ValueError("Non-finite adapter gradient")
        return gradient

    for name, parameter in trainer.model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError(f"Unexpected trainable base parameter: {name}")
            initial[name] = parameter.detach().cpu().clone()
            parameter.register_hook(check_gradient)
    if not initial:
        raise ValueError("No trainable LoRA parameters")
    if resume_directory and resume["global_step"] == settings.max_steps:
        from types import SimpleNamespace

        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(
            trainer.model, load_file(str(resume_directory / "adapter_model.safetensors"))
        )
        result = SimpleNamespace(global_step=settings.max_steps)
        ctx.emit(
            "training_finalized_from_checkpoint",
            global_step=settings.max_steps,
            checkpoint=resume_id,
        )
    else:
        result = trainer.train(
            resume_from_checkpoint=str(resume_directory) if resume_directory else None
        )
    delta = sum(
        (parameter.detach().cpu().float() - initial[name].float()).abs().sum().item()
        for name, parameter in trainer.model.named_parameters()
        if name in initial
    )
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Training did not produce a finite nonzero adapter update")
    adapter = ctx.work / "adapter"
    trainer.model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    parent = {"id": parent_id, "manifest_sha": ctx.store.artifact(parent_id)["manifest_sha"]}
    adapter_id = ctx.artifacts.publish(
        ctx.id,
        "lora_adapter",
        adapter,
        {
            "parent": parent,
            "dataset": ctx.prepared()["dataset"],
            "spec_sha": fingerprint(json.loads(ctx.run["spec"])),
            "prompt_profile": ctx.spec.model.prompt_profile,
        },
    )
    output = {
        "adapter": adapter_id,
        "checkpoints": checkpoint_ids,
        "global_step": result.global_step,
        "adapter_delta_l1": delta,
        "tokens_seen": counters["tokens"],
        "supervised_tokens_seen": counters["supervised"],
        "elapsed_seconds": time.monotonic() - started,
        "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    }
    ctx.emit("training_verified", **output)
    return output

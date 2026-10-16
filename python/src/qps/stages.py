"""Worker phase implementations shared by the smoke and CUDA recipes."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from qps.artifacts import Artifacts, file_manifest, verify_artifact
from qps.contracts import ArtifactPublisher, EventSink, PhaseOutputs
from qps.data import final_answer, prepare_records
from qps.specs import RunSpec
from qps.store import Store
from qps.util import atomic_json, atomic_text, canonical, digest, fingerprint


def environment():
    import torch

    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "trl",
            "peft",
            "datasets",
            "accelerate",
            "tokenizers",
        )
    }
    source_root = Path(__file__).parent
    code = {p.name: digest(p) for p in sorted(source_root.glob("*.py"))}
    lock = source_root.parents[1] / "uv.lock"
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": sys.version,
        "packages": packages,
        "code_sha": fingerprint(code),
        "git_revision": os.environ.get("QPS_CODE_REVISION") or revision,
        "lock_sha": os.environ.get("QPS_LOCK_SHA") or (digest(lock) if lock.exists() else None),
        "image_digest": os.environ.get("QPS_IMAGE_DIGEST") or None,
        "cuda": torch.version.cuda,
    }


class Context:
    def __init__(
        self,
        state: Path,
        run_id: str,
        attempt: int,
        *,
        event_sink: EventSink | None = None,
        artifact_publisher: ArtifactPublisher | None = None,
        phase_outputs: PhaseOutputs | None = None,
    ):
        self.store = Store(state)
        self.run = self.store.run(run_id)
        self.id = self.run["id"]
        self.attempt = attempt
        self.spec = RunSpec.model_validate_json(self.run["spec"]).spec
        self.artifacts = artifact_publisher or Artifacts(self.store, self.spec.artifacts.root)
        self.events = event_sink or self.store
        self.phases = phase_outputs or self.store
        self.work = self.store.directory(self.id) / "attempts" / str(attempt)
        self.work.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **payload):
        self.events.emit(self.id, self.attempt, kind, payload)

    def output(self, phase):
        result = self.phases.output(self.id, phase)
        if result is None:
            raise ValueError(f"Missing committed output for phase {phase}")
        return result

    def artifact(self, artifact_id):
        return verify_artifact(self.store, artifact_id)

    def prepared(self):
        return self.output("prepare")

    def records(self):
        directory, _ = self.artifact(self.prepared()["dataset"])
        return {
            key: [
                json.loads(line) for line in (directory / f"{key}.jsonl").read_text().splitlines()
            ]
            for key in ("train", "validation")
        }


def prepare(ctx: Context):
    from transformers import AutoConfig

    from qps.ml import configure, tokenizer_from

    device, _ = configure(ctx.spec)  # Fail before downloads on an incompatible machine.
    source = ctx.spec.model
    if source.source == "huggingface":
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().model_info(source.id, revision=source.revision).sha
        if not revision:
            raise ValueError("Could not resolve model revision")
        model_directory = Path(
            snapshot_download(
                source.id,
                revision=revision,
                allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"],
            )
        )
    else:
        model_directory = Path(source.id)
        revision = fingerprint(file_manifest(model_directory))
    config = AutoConfig.from_pretrained(
        model_directory, trust_remote_code=False, local_files_only=True
    )
    if config.model_type != "qwen3":
        raise ValueError("Only dense Qwen3 snapshots are supported")
    tokenizer = tokenizer_from(model_directory)
    if ctx.spec.training.max_sequence_length > config.max_position_embeddings:
        raise ValueError("Training sequence budget exceeds model context")
    parent_metadata = {
        "source": source.model_dump(),
        "revision": revision,
        "prompt_profile": source.prompt_profile,
        "template_sha": fingerprint(tokenizer.chat_template),
        "source_files_sha": fingerprint(file_manifest(model_directory)),
    }
    parent = ctx.artifacts.publish(ctx.id, "base_model", model_directory, parent_metadata)
    dataset = ctx.spec.dataset
    if dataset.source == "huggingface":
        from datasets import load_dataset
        from huggingface_hub import HfApi

        dataset_revision = HfApi().dataset_info(dataset.id, revision=dataset.revision).sha
        if not dataset_revision:
            raise ValueError("Could not resolve dataset revision")
        rows = load_dataset(
            dataset.id, dataset.configuration, revision=dataset_revision, split=dataset.source_split
        )
    else:
        data_file = Path(dataset.id)
        dataset_revision = digest(data_file)
        rows = [json.loads(line) for line in data_file.read_text().splitlines() if line.strip()]
    train, validation, split = prepare_records(
        rows,
        tokenizer,
        dataset,
        ctx.spec.training.max_sequence_length,
    )
    if any(
        len(row["prompt_ids"]) + ctx.spec.evaluation.max_new_tokens > config.max_position_embeddings
        for row in validation
    ):
        raise ValueError("Validation prompt plus output budget exceeds model context")
    data_directory = ctx.work / "prepared-data"
    data_directory.mkdir()
    for key, records in (("train", train), ("validation", validation)):
        atomic_text(
            data_directory / f"{key}.jsonl", "".join(canonical(row) + "\n" for row in records)
        )
    atomic_json(data_directory / "split.json", split)
    data_metadata = {"source": dataset.model_dump(), "revision": dataset_revision, **split}
    data_id = Artifacts(ctx.store, ctx.store.root / "datasets").publish(
        ctx.id,
        "dataset",
        data_directory,
        data_metadata,
    )
    resolved = {
        "parent": parent,
        "dataset": data_id,
        "parent_revision": revision,
        "dataset_revision": dataset_revision,
        "environment": environment(),
        "device": device,
        "spec_sha": fingerprint(json.loads(ctx.run["spec"])),
        "host_disk_free_bytes": shutil.disk_usage(ctx.store.root).free,
    }
    ctx.store.update(ctx.id, resolved=canonical(resolved))
    atomic_json(ctx.store.directory(ctx.id) / "resolved.json", resolved)
    ctx.emit(
        "data_prepared",
        train=len(train),
        validation=len(validation),
        rejected=len(split["rejected"]),
        parent=parent,
        dataset=data_id,
    )
    return resolved


def evaluate_artifact(ctx: Context, artifact_id: str):
    from qps.ml import generate, load_model, tokenizer_from

    directory, _ = ctx.artifact(artifact_id)
    tokenizer = tokenizer_from(directory)
    model = load_model(directory, ctx.spec)
    predictions = []
    validation = ctx.records()["validation"]
    for index, row in enumerate(validation):
        result = generate(model, tokenizer, row["prompt_ids"], ctx.spec.evaluation.max_new_tokens)
        parsed = final_answer(result["text"])
        predictions.append(
            {
                "problem_id": row["problem_id"],
                "reference": row["reference_answer"],
                **result,
                "parsed": str(parsed) if parsed is not None else None,
                "correct": parsed is not None and str(parsed) == row["reference_answer"],
            }
        )
        ctx.emit(
            "evaluation_progress",
            artifact_id=artifact_id,
            completed=index + 1,
            total=len(validation),
        )
    correct = sum(row["correct"] for row in predictions)
    report = {
        "artifact_id": artifact_id,
        "dataset_id": ctx.prepared()["dataset"],
        "profile": ctx.spec.evaluation.model_dump(),
        "predictions": predictions,
        "correct": correct,
        "count": len(predictions),
        "accuracy": correct / len(predictions),
        "parse_failures": sum(row["parsed"] is None for row in predictions),
        "truncations": sum(row["finish_reason"] == "length" for row in predictions),
        "generation_failures": 0,
    }
    return report


def publish_report(ctx, name, report, kind="evaluation_report"):
    directory = ctx.work / name
    directory.mkdir(exist_ok=True)
    atomic_json(directory / "report.json", report)
    return ctx.artifacts.publish(ctx.id, kind, directory, {"report": name})


def baseline(ctx: Context):
    report = evaluate_artifact(ctx, ctx.prepared()["parent"])
    return {"report": publish_report(ctx, "baseline", report)}


def export(ctx: Context):
    import torch
    from peft import PeftModel

    from qps.ml import PARITY_TOLERANCES, load_model, probe_logits, tokenizer_from

    parent_id = ctx.prepared()["parent"]
    parent_directory, _ = ctx.artifact(parent_id)
    adapter_id = ctx.output("training")["adapter"]
    adapter_directory, manifest = ctx.artifact(adapter_id)
    expected_parent = {
        "id": parent_id,
        "manifest_sha": ctx.store.artifact(parent_id)["manifest_sha"],
    }
    if manifest["metadata"]["parent"] != expected_parent:
        raise ValueError("Adapter parent identity is incompatible with this exact base checkpoint")
    tokenizer = tokenizer_from(parent_directory)
    # Keep a full-precision master export. Rounding each merged matrix into BF16
    # accumulates through the decoder and needlessly weakens the parity guarantee.
    parent = load_model(parent_directory, ctx.spec, dtype_override=torch.float32)
    adapted = PeftModel.from_pretrained(parent, adapter_directory).eval()
    probes = [row["input_ids"] for row in ctx.records()["validation"][:3]]
    reference = [probe_logits(adapted, ids) for ids in probes]
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    destination = ctx.work / "merged"
    merged.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    merged.generation_config.save_pretrained(destination)
    del adapted, parent, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded = load_model(destination, ctx.spec, dtype_override=torch.float32)
    tolerance = PARITY_TOLERANCES["fp32"]
    maximum = 0.0
    for ids, expected in zip(probes, reference, strict=True):
        actual = probe_logits(reloaded, ids)
        torch.testing.assert_close(actual, expected, **tolerance)
        maximum = max(maximum, (actual - expected).abs().max().item())
    atomic_json(
        destination / "prompt-profile.json",
        {
            "id": ctx.spec.model.prompt_profile,
            "enable_thinking": False,
            "template_sha": fingerprint(tokenizer.chat_template),
        },
    )
    parity = {
        "dtype": "fp32",
        "training_dtype": ctx.spec.training.dtype,
        "inference_dtype": ctx.spec.training.dtype,
        "tolerances": tolerance,
        "max_absolute_logit_difference": maximum,
        "probes": len(probes),
    }
    model_id = ctx.artifacts.publish(
        ctx.id,
        "merged_model",
        destination,
        {
            "parent": expected_parent,
            "adapter_id": adapter_id,
            "parity": parity,
            "dataset_id": ctx.prepared()["dataset"],
            "prompt_profile": ctx.spec.model.prompt_profile,
        },
    )
    ctx.emit("export_verified", artifact_id=model_id, **parity)
    return {"model": model_id, "parity": parity}


def evaluation(ctx: Context):
    baseline_directory, _ = ctx.artifact(ctx.output("baseline")["report"])
    baseline_report = json.loads((baseline_directory / "report.json").read_text())
    candidate = evaluate_artifact(ctx, ctx.output("export")["model"])
    candidate_id = publish_report(ctx, "candidate", candidate)
    comparison = {
        "baseline": baseline_report,
        "candidate": candidate,
        "accuracy_delta": candidate["accuracy"] - baseline_report["accuracy"],
        "training": ctx.output("training"),
        "parity": ctx.output("export")["parity"],
        "provenance": ctx.prepared(),
        "metrics": [e for e in ctx.store.events(ctx.id) if e["kind"] == "train_step"],
        "quality_claim": "Systems smoke result; no improvement or statistical significance implied.",
        "total_active_seconds": ctx.store.run(ctx.id)["elapsed_seconds"],
        "durable_artifact_bytes": sum(
            sum(
                item["size"]
                for item in json.loads((Path(record["path"]) / "manifest.json").read_text())[
                    "files"
                ].values()
            )
            for record in ctx.store.artifacts(ctx.id)
        ),
    }
    report_id = publish_report(ctx, "comparison", comparison, "comparison_report")
    atomic_json(ctx.store.directory(ctx.id) / "comparison.json", comparison)
    ctx.emit(
        "comparison_ready", baseline=baseline_report["accuracy"], candidate=candidate["accuracy"]
    )
    return {"report": candidate_id, "comparison": report_id}

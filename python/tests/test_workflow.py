import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from qps.artifacts import Artifacts, verify_artifact
from qps.pipeline import coordinate
from qps.specs import RunSpec
from qps.stages import Context, baseline, evaluation, export, prepare
from qps.store import Store
from qps.util import process_identity

pytestmark = pytest.mark.integration


def test_complete_cpu_workflow(successful):
    store, report = successful
    assert report["status"] == "passed" and report["gpu_validated"] is False
    assert report["restart_verified"] and report["http_evaluation"]["http_offline_parity"]
    assert report["training"]["adapter_delta_l1"] > 0
    assert report["parity"]["max_absolute_logit_difference"] < 1e-5
    assert store.run(report["run_id"])["status"] == "succeeded"
    assert store.deployment(report["name"])["state"] == "stopped"


def test_export_preserves_fp32_master_when_execution_precision_is_bf16(
    successful, tmp_path, monkeypatch
):
    import torch
    from safetensors import safe_open

    store, report = successful
    run = store.run(report["run_id"])
    ctx = Context(store.root, run["id"], run["attempt"])
    ctx.work = tmp_path / "export-master"
    ctx.work.mkdir()
    # Exercise BF16 execution selection without requiring CUDA in the CPU suite.
    monkeypatch.setattr("qps.ml.configure", lambda spec: ("cpu", torch.bfloat16))
    output = export(ctx)
    directory, manifest = verify_artifact(store, output["model"])
    with safe_open(directory / "model.safetensors", framework="pt") as weights:
        assert {weights.get_slice(key).get_dtype() for key in weights.keys()} == {"F32"}
    assert output["parity"]["dtype"] == "fp32"
    assert manifest["metadata"]["parity"] == output["parity"]
    assert output["parity"]["max_absolute_logit_difference"] < 1e-5


def test_interrupted_training_resumes_optimizer_and_matches_uninterrupted(successful, tmp_path):
    import torch
    from safetensors.torch import load_file

    from qps.training import train

    reference_store, report = successful
    spec = RunSpec.model_validate_json(reference_store.run(report["run_id"])["spec"])
    spec.metadata.name = "resumed-run"
    spec.spec.artifacts.root = tmp_path / "artifacts"
    store = Store(tmp_path / "state")
    run = store.create_run(spec)
    attempt = store.begin_attempt(run["id"], process_identity())
    ctx = Context(store.root, run["id"], attempt)
    for phase, function in (("prepare", prepare), ("baseline", baseline)):
        store.phase(run["id"], attempt, phase, "running")
        store.phase(run["id"], attempt, phase, "succeeded", function(ctx))
    original = ctx.emit

    def interrupt_at_checkpoint(kind, **payload):
        original(kind, **payload)
        if kind == "checkpoint_saved" and payload["global_step"] == 2:
            raise RuntimeError("Simulated interruption after durable checkpoint")

    ctx.emit = interrupt_at_checkpoint
    store.phase(run["id"], attempt, "training", "running")
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        train(ctx)
    store.phase(run["id"], attempt, "training", "failed")
    store.finish(run["id"], attempt, "interrupted", "test interruption")
    checkpoint = [
        event for event in store.events(run["id"]) if event["kind"] == "checkpoint_saved"
    ][-1]
    coordinate(
        store,
        run["id"],
        action="resume",
        checkpoint=checkpoint["payload"]["artifact_id"],
        verbose=False,
    )
    metrics = [event for event in store.events(run["id"]) if event["kind"] == "train_step"]
    assert [(e["attempt"], e["payload"]["global_step"]) for e in metrics] == [
        (1, 1),
        (1, 2),
        (2, 3),
        (2, 4),
    ]
    assert metrics[-1]["payload"]["tokens_seen"] == report["training"]["tokens_seen"]
    restored = [event for event in store.events(run["id"]) if event["kind"] == "resume_verified"]
    assert restored[0]["payload"]["optimizer_min_step"] == 2
    reference_path, _ = verify_artifact(reference_store, report["training"]["adapter"])
    resumed_path, _ = verify_artifact(store, store.output(run["id"], "training")["adapter"])
    expected = load_file(str(reference_path / "adapter_model.safetensors"))
    actual = load_file(str(resumed_path / "adapter_model.safetensors"))
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-7)


def test_failed_evaluation_retries_without_training(successful, monkeypatch):
    store, report = successful
    run_id = report["run_id"]
    training = store.output(run_id, "training")
    attempt = store.begin_attempt(run_id, process_identity())
    store.phase(run_id, attempt, "evaluation", "running")

    def unavailable(*args):
        raise RuntimeError("Evaluator unavailable")

    with monkeypatch.context() as patch:
        patch.setattr("qps.stages.evaluate_artifact", unavailable)
        with pytest.raises(RuntimeError, match="Evaluator unavailable"):
            evaluation(Context(store.root, run_id, attempt))
    store.phase(run_id, attempt, "evaluation", "failed")
    store.finish(run_id, attempt, "failed", "Evaluator unavailable")
    before = len([event for event in store.events(run_id) if event["kind"] == "train_step"])
    coordinate(store, run_id, action="retry-evaluation", verbose=False)
    assert store.output(run_id, "training") == training
    assert len([e for e in store.events(run_id) if e["kind"] == "train_step"]) == before
    assert store.run(run_id)["status"] == "succeeded"


def test_incompatible_adapter_parent_rejected(successful, tmp_path):
    store, report = successful
    run = store.run(report["run_id"])
    spec = RunSpec.model_validate_json(run["spec"]).spec
    adapter, manifest = verify_artifact(store, report["training"]["adapter"])
    bad = Artifacts(store, spec.artifacts.root).publish(
        run["id"],
        "lora_adapter",
        adapter,
        {
            **manifest["metadata"],
            "parent": {"id": "wrong", "manifest_sha": "wrong"},
        },
    )
    ctx = Context(store.root, run["id"], run["attempt"])
    output = ctx.output
    ctx.output = lambda phase: {"adapter": bad} if phase == "training" else output(phase)
    with pytest.raises(ValueError, match="incompatible"):
        export(ctx)


def test_http_validation_admission_and_generation_error_recovery(successful, monkeypatch):
    from fastapi.testclient import TestClient

    import qps.ml
    from qps.server import create_app

    store, report = successful
    original = qps.ml.generate
    entered = threading.Event()
    release = threading.Event()
    gate = {"enabled": False, "fail": False}

    def controlled(*args, **kwargs):
        if gate["enabled"]:
            entered.set()
            assert release.wait(5)
        if gate["fail"]:
            raise RuntimeError("injected generation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(qps.ml, "generate", controlled)
    app = create_app(store, report["artifact_id"], "test-model")
    body = {"model": "test-model", "prompt": "hello", "max_tokens": 8}
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/readyz").json()["artifact_id"] == report["artifact_id"]
        assert client.post("/v1/completions", json={**body, "temperature": 0.7}).status_code == 422
        assert client.post("/v1/completions", json={**body, "tools": []}).status_code == 422
        assert client.post("/v1/completions", json={**body, "model": "wrong"}).status_code == 404
        assert client.post("/v1/completions", json={**body, "max_tokens": 512}).status_code == 422
        gate["enabled"] = True
        with ThreadPoolExecutor() as executor:
            pending = executor.submit(client.post, "/v1/completions", json=body)
            assert entered.wait(5)
            assert client.post("/v1/completions", json=body).status_code == 429
            gate["enabled"] = False
            release.set()
            assert pending.result(timeout=5).status_code == 200
        gate["fail"] = True
        assert client.post("/v1/completions", json=body).status_code == 500
        failed_stream = client.post("/v1/completions", json={**body, "stream": True})
        assert '"error"' in failed_stream.text and "data: [DONE]" in failed_stream.text
        gate["fail"] = False
        assert client.post("/v1/completions", json=body).status_code == 200


def test_gradient_accumulation_matches_a_full_variable_length_batch(successful, tmp_path):
    from qps.training import train

    reference_store, report = successful
    original_spec = RunSpec.model_validate_json(reference_store.run(report["run_id"])["spec"])
    rows = [
        json.loads(line) for line in Path(original_spec.spec.dataset.id).read_text().splitlines()
    ]
    for index, row in enumerate(rows):
        row["answer"] = ("extra explanation " * (index % 3)) + row["answer"]
    source = tmp_path / "variable-length.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    measurements = []
    for batch, accumulation in ((2, 1), (1, 2)):
        spec = original_spec.model_copy(deep=True)
        spec.metadata.name = f"batch-{batch}"
        spec.spec.dataset.id = str(source)
        spec.spec.artifacts.root = tmp_path / f"artifacts-{batch}"
        spec.spec.training.max_steps = 1
        spec.spec.training.checkpoint_every_steps = 1
        spec.spec.training.micro_batch_size = batch
        spec.spec.training.gradient_accumulation_steps = accumulation
        store = Store(tmp_path / f"state-{batch}")
        run = store.create_run(spec)
        attempt = store.begin_attempt(run["id"], process_identity())
        ctx = Context(store.root, run["id"], attempt)
        store.phase(run["id"], attempt, "prepare", "succeeded", prepare(ctx))
        train(ctx)
        metrics = [e["payload"] for e in store.events(run["id"]) if e["kind"] == "train_step"]
        measurements.append(metrics[0])
    assert measurements[0]["supervised_tokens_seen"] == measurements[1]["supervised_tokens_seen"]
    assert measurements[0]["grad_norm"] == pytest.approx(measurements[1]["grad_norm"], rel=1e-5)
    assert measurements[0]["loss"] == pytest.approx(measurements[1]["loss"], abs=1e-4)


def test_completed_checkpoint_can_finalize_after_lost_training_output(successful):
    store, report = successful
    original_steps = len([e for e in store.events(report["run_id"]) if e["kind"] == "train_step"])
    checkpoint = report["training"]["checkpoints"][-1]
    coordinate(store, report["run_id"], action="resume", checkpoint=checkpoint, verbose=False)
    assert store.run(report["run_id"])["status"] == "succeeded"
    assert (
        len([e for e in store.events(report["run_id"]) if e["kind"] == "train_step"])
        == original_steps
    )
    assert any(
        e["kind"] == "training_finalized_from_checkpoint" for e in store.events(report["run_id"])
    )


def test_stream_disconnect_releases_capacity(successful):
    import time

    import httpx

    from qps import deploy

    store, report = successful
    deployment = deploy.create(store, report["run_id"], "disconnect-test", port=0)
    body = {"model": "disconnect-test", "prompt": "hello", "max_tokens": 256}
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            with client.stream(
                "POST", deployment["url"] + "/v1/completions", json={**body, "stream": True}
            ) as response:
                response.raise_for_status()
                next(response.iter_lines())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                result = client.post(
                    deployment["url"] + "/v1/completions", json={**body, "max_tokens": 1}
                )
                if result.status_code == 200:
                    break
                assert result.status_code == 429
                time.sleep(0.05)
            else:
                pytest.fail("Disconnected generation retained serving capacity")
    finally:
        deploy.stop(store, "disconnect-test")

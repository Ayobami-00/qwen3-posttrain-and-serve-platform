import time
import uuid
from types import SimpleNamespace

import boto3
import pytest
from moto import mock_aws
from sqlalchemy import insert, update
from test_hosted_storage import registry as registry

from qps.contracts import DeploymentSubmission, HostedSubmission
from qps.hosted.database import (
    Conflict,
    Fenced,
    allocations,
    artifacts,
    attempts,
    cost_ledger,
    deployments,
    models,
    operations,
    runs,
    uploads,
)
from qps.hosted.deletion import Deleter
from qps.hosted.objects import ObjectStore

REPORT = {
    "baseline": {"accuracy": 0.25, "correct": 1, "count": 4, "parse_failures": 0, "truncations": 0},
    "candidate": {"accuracy": 0.5, "correct": 2, "count": 4, "parse_failures": 0, "truncations": 0},
    "accuracy_delta": 0.25,
}


def published(registry):
    repo, run, submission = registry
    ids = {
        kind: "art-" + uuid.uuid4().hex
        for kind in (
            "base_model",
            "dataset",
            "lora_adapter",
            "merged_model",
            "comparison_report",
            "training_checkpoint",
        )
    }
    outputs = {
        "prepare": {"parent": ids["base_model"], "dataset": ids["dataset"]},
        "baseline": {},
        "training": {"adapter": ids["lora_adapter"]},
        "export": {"model": ids["merged_model"]},
        "evaluation": {"comparison": ids["comparison_report"]},
    }
    with repo.transaction() as c:
        c.execute(
            update(runs).values(
                resolved={"run": submission.run.model_dump(mode="json", by_alias=True)},
                status="succeeded",
            )
        )
        c.execute(update(attempts).values(outputs=outputs, status="succeeded"))
        for kind, identity in ids.items():
            c.execute(
                insert(artifacts).values(
                    id=identity,
                    run_id=run["id"],
                    attempt=1,
                    kind=kind,
                    manifest_sha=identity,
                    manifest={"metadata": {}},
                    objects={},
                    size=1,
                    created=time.time(),
                )
            )
            repo.add_reference(c, run["id"], identity, "experiment")
    objects = SimpleNamespace(
        read_json=lambda _: REPORT,
        deletion_tombstones=lambda plan: None,
        ledger=lambda allocation: None,
        purge_prefix=lambda *a, **k: None,
        key=lambda *p: "test/" + "/".join(p),
    )
    model = repo.promote_model(run["id"], objects)
    return repo, run, model, ids, objects


def test_experiment_deletion_preserves_published_model_and_serving_configuration(registry):
    repo, run, model, ids, objects = published(registry)
    dep = repo.deployment_submit(
        DeploymentSubmission(
            name="survives",
            model_version_id=model["id"],
            compute_profile="cpu-fixture",
            idempotency_key="serve-0001",
        )
    )
    backup = repo.export_metadata()
    op = repo.start_deletion("experiment", run["id"])
    assert ids["merged_model"] in op["plan"]["preservedArtifacts"]
    assert ids["training_checkpoint"] in op["plan"]["artifacts"]
    with pytest.raises(Fenced):
        repo.action(run["id"], "resume")
    Deleter(repo, objects).reconcile(op["id"])
    assert repo.one(operations, op["id"])["state"] == "completed"
    with pytest.raises(KeyError):
        repo.one(runs, run["id"])
    assert repo.one(models, model["id"])["snapshot"]["evaluation"] == REPORT
    assert repo.one(deployments, dep["id"])["model_version_id"] == model["id"]
    assert repo.one(artifacts, ids["merged_model"])["run_id"] == run["id"]
    repo.restore_metadata(backup)
    assert repo.list(runs) == []
    assert repo.one(models, model["id"])["status"] == "ready"
    assert not any(r["kind"] == "training_checkpoint" for r in repo.list(artifacts))


def test_model_deletion_cascades_deployments_but_keeps_experiment_and_shared_files(registry):
    repo, run, model, ids, objects = published(registry)
    dep = repo.deployment_submit(
        DeploymentSubmission(
            name="cascade",
            artifact_id=ids["merged_model"],
            compute_profile="cpu-fixture",
            idempotency_key="serve-0002",
        )
    )
    op = repo.start_deletion("model", model["id"])
    assert dep["id"] in op["plan"]["deployments"]
    # Publication transfers weight ownership, while the experiment retains shared inputs.
    assert set(op["plan"]["artifacts"]) == {ids["lora_adapter"], ids["merged_model"]}
    with pytest.raises(Conflict):
        repo.deployment_submit(
            DeploymentSubmission(
                name="late",
                model_version_id=model["id"],
                compute_profile="cpu-fixture",
                idempotency_key="serve-late",
            )
        )
    Deleter(repo, objects).reconcile(op["id"])
    assert repo.list(deployments) == [] and repo.list(models) == []
    assert repo.one(runs, run["id"])
    assert repo.one(artifacts, ids["dataset"])


def test_cost_and_late_uploads_survive_restarted_deletion_and_restore(registry):
    repo, run, _ = registry
    alloc = repo.reserve(run["id"], "training", {"hourlyRate": 0, "gpuType": "cpu"})
    with repo.transaction() as c:
        c.execute(
            insert(uploads).values(
                id="late-upload",
                allocation_id=alloc["id"],
                expires=time.time() + 100,
                manifest={},
                state="staging",
            )
        )
    op = repo.start_deletion("experiment", run["id"])
    deleter = Deleter(
        repo,
        SimpleNamespace(
            deletion_tombstones=lambda p: None,
            ledger=lambda a: None,
            purge_prefix=lambda *a, **k: None,
            key=lambda *p: "test/" + "/".join(p),
        ),
    )
    deleter.reconcile(op["id"])
    assert repo.one(operations, op["id"])["phase"] == "stopping"
    with repo.connection() as connection, pytest.raises(Fenced):
        repo.assert_live(connection, alloc)
    repo.observe_allocation(alloc["id"], {"state": "terminated"})
    with repo.transaction() as c:
        c.execute(update(cost_ledger).where(cost_ledger.c.id == alloc["id"]).values(usd=0.83))
    backup = repo.export_metadata()
    deleter.reconcile(op["id"])
    assert repo.one(operations, op["id"])["phase"] == "waiting-for-upload-expiry"
    with repo.transaction() as c:
        plan = {**op["plan"], "uploadsUntil": 0}
        c.execute(update(operations).values(plan=plan))
        c.execute(update(uploads).values(expires=0))
    Deleter(repo, deleter.objects).reconcile(op["id"])
    assert repo.list(allocations) == []
    assert repo.budget()["estimatedSpentUsd"] == 0.83
    repo.restore_metadata(backup)
    assert repo.list(allocations) == [] and repo.budget()["estimatedSpentUsd"] == 0.83


def test_s3_purge_removes_versions_delete_markers_and_multipart_uploads(registry):
    repo, _, _ = registry
    with mock_aws():
        client = boto3.client("s3", region_name="eu-west-2")
        client.create_bucket(
            Bucket=repo.settings.bucket,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
        )
        client.put_bucket_versioning(
            Bucket=repo.settings.bucket, VersioningConfiguration={"Status": "Enabled"}
        )
        key = "test/artifacts/art-owned/files/model"
        for data in (b"first", b"second"):
            client.put_object(Bucket=repo.settings.bucket, Key=key, Body=data)
        client.delete_object(Bucket=repo.settings.bucket, Key=key)
        client.create_multipart_upload(Bucket=repo.settings.bucket, Key=key + "-partial")
        client.put_object(
            Bucket=repo.settings.bucket, Key="test/artifacts/other/files/model", Body=b"keep"
        )
        store = ObjectStore(repo.settings, repo, client)
        store.purge_prefix("test/artifacts/art-owned/")
        page = client.list_object_versions(
            Bucket=repo.settings.bucket, Prefix="test/artifacts/art-owned/"
        )
        assert not page.get("Versions") and not page.get("DeleteMarkers")
        assert not client.list_multipart_uploads(Bucket=repo.settings.bucket).get("Uploads")
        assert (
            client.get_object(Bucket=repo.settings.bucket, Key="test/artifacts/other/files/model")[
                "Body"
            ].read()
            == b"keep"
        )
        with pytest.raises(ValueError):
            store.purge_prefix("test/")


def test_four_gpu_units_admit_mixed_training_and_serving_without_losing_spend(registry):
    from pathlib import Path

    from qps.specs import load_spec

    repo, _, _ = registry
    # Fixture is older in FIFO; cancel it before submitting GPU workloads.
    repo.action(registry[1]["id"], "cancel")
    two = []
    for index in range(2):
        spec = load_spec(Path(__file__).resolve().parents[2] / "examples/local/gsm8k-lora.yaml")
        spec.metadata.name = f"fsdp-{index}"
        spec.spec.execution.parallelism = "fsdp2"
        spec.spec.execution.gpu_count = 2
        spec.spec.execution.max_wall_time_seconds = 2700
        req = HostedSubmission(
            run=spec, compute_profile="runpod-qwen3-fsdp2", idempotency_key=f"two-gpu-{index}"
        )
        two.append(repo.submit(req, {}))
    first = repo.reserve(
        two[0]["id"], "training", {"hourlyRate": 1.18, "gpuType": "NVIDIA A40", "gpuCount": 2}
    )
    repo.reserve(
        two[1]["id"], "training", {"hourlyRate": 1.18, "gpuType": "NVIDIA A40", "gpuCount": 2}
    )
    with repo.transaction() as c:
        for n in range(2):
            c.execute(
                insert(deployments).values(
                    id=f"dep-single-{n}",
                    name=f"single-{n}",
                    generation=1,
                    status="queued",
                    desired="run",
                    created=time.time() + n,
                    spec={"computeProfile": "runpod-qwen3-small", "maxRuntimeSeconds": 900},
                )
            )
    with pytest.raises(Conflict, match="slots"):
        repo.reserve(
            "dep-single-0",
            "deployment",
            {"hourlyRate": 0.59, "gpuType": "NVIDIA A40", "gpuCount": 1},
        )
    repo.observe_allocation(first["id"], {"state": "terminated"})
    for n in range(2):
        repo.reserve(
            f"dep-single-{n}",
            "deployment",
            {"hourlyRate": 0.59, "gpuType": "NVIDIA A40", "gpuCount": 1},
        )
    assert repo.budget()["activeGpuUnits"] == 4
    assert repo.budget()["reservedUsd"] > 0
    assert repo.budget()["estimatedSpentUsd"] > 0


def test_fresh_database_restore_merges_durable_s3_tombstones_and_costs(registry):
    from qps.hosted.database import Repository

    repo, run, _ = registry
    allocation = repo.reserve(run["id"], "training", {"hourlyRate": 0, "gpuType": "cpu"})
    backup = repo.export_metadata()
    op = repo.start_deletion("experiment", run["id"])
    repo.observe_allocation(allocation["id"], {"state": "terminated"})
    with repo.transaction() as c:
        c.execute(update(cost_ledger).values(usd=0.83, gpu_allocations=1))
    with mock_aws():
        client = boto3.client("s3", region_name="eu-west-2")
        client.create_bucket(
            Bucket=repo.settings.bucket,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
        )
        store = ObjectStore(repo.settings, repo, client)
        store.deletion_tombstones(op["plan"])
        store.ledger(repo.one(allocations, allocation["id"]))
        fresh = Repository(repo.settings, create=True)
        fresh.restore_metadata(backup, recovery_ledger=store.recovery_ledger())
        assert fresh.list(runs) == [] and fresh.list(allocations) == []
        assert fresh.budget()["estimatedSpentUsd"] == 0.83
        assert fresh.list(cost_ledger)[0]["gpu_allocations"] == 1


def test_concurrent_model_deletions_collect_the_last_shared_inputs(registry):
    repo, run, first, ids, objects = published(registry)
    second = {
        **first,
        "id": "model-" + uuid.uuid4().hex,
        "artifact_id": "art-" + uuid.uuid4().hex,
        "adapter_id": "art-" + uuid.uuid4().hex,
    }
    with repo.transaction() as c:
        c.execute(insert(models).values(**second))
        for identity in (second["artifact_id"], second["adapter_id"]):
            c.execute(
                insert(artifacts).values(
                    id=identity,
                    run_id=run["id"],
                    kind="merged_model" if identity == second["artifact_id"] else "lora_adapter",
                    manifest={},
                    objects={},
                    created=time.time(),
                )
            )
        for identity in (
            second["artifact_id"],
            second["adapter_id"],
            ids["base_model"],
            ids["dataset"],
        ):
            repo.add_reference(c, second["id"], identity, "model")
    deleter = Deleter(repo, objects)
    experiment = repo.start_deletion("experiment", run["id"])
    deleter.reconcile(experiment["id"])
    a = repo.start_deletion("model", first["id"])
    b = repo.start_deletion("model", second["id"])
    assert (
        ids["dataset"] in a["plan"]["preservedArtifacts"]
        and ids["dataset"] in b["plan"]["preservedArtifacts"]
    )
    deleter.reconcile(a["id"])
    deleter.reconcile(b["id"])
    pending = repo.one(operations, b["id"])
    assert pending["phase"] == "collecting-shared-inputs"
    deleter.reconcile(pending["plan"]["childOperationId"])
    deleter.reconcile(b["id"])
    assert repo.one(operations, b["id"])["state"] == "completed"
    assert repo.list(artifacts) == []


def test_inflight_copy_is_fenced_and_unregistered_versions_are_purged(registry, monkeypatch):
    import hashlib
    from datetime import UTC, datetime

    repo, run, _ = registry
    repo.settings.upload_url_seconds = 1
    allocation = repo.reserve(run["id"], "training", {"hourlyRate": 0, "gpuType": "cpu"})
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-2")
        s3.create_bucket(
            Bucket=repo.settings.bucket,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
        )
        s3.put_bucket_versioning(
            Bucket=repo.settings.bucket, VersioningConfiguration={"Status": "Enabled"}
        )
        store = ObjectStore(repo.settings, repo, s3)
        body = b"in-flight model"
        manifest = {
            "version": 1,
            "id": "art-" + uuid.uuid4().hex,
            "run_id": run["id"],
            "kind": "training_checkpoint",
            "created": datetime.now(UTC).isoformat(),
            "metadata": {"global_step": 25},
            "files": {
                "adapter.safetensors": {
                    "size": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            },
        }
        upload = store.start(allocation, manifest)
        s3.put_object(
            Bucket=store.bucket,
            Key=store.key("staging", allocation["id"], upload["uploadId"], "adapter.safetensors"),
            Body=body,
        )
        original = s3.copy_object
        started = []

        def delayed_copy(**kwargs):
            started.append(repo.start_deletion("experiment", run["id"]))
            # The request was already authorized when deletion fenced the worker.
            return original(**kwargs)

        monkeypatch.setattr(s3, "copy_object", delayed_copy)
        with pytest.raises(Fenced):
            store.commit(allocation, upload["uploadId"])
        op = started[0]
        assert manifest["id"] in op["plan"]["artifacts"]
        assert repo.list(artifacts) == []
        repo.observe_allocation(allocation["id"], {"state": "terminated"})
        deleter = Deleter(repo, store)
        deleter.reconcile(op["id"])
        assert repo.one(operations, op["id"])["phase"] == "waiting-for-upload-expiry"
        with repo.transaction() as c:
            c.execute(update(uploads).values(expires=0))
            c.execute(update(operations).values(plan={**op["plan"], "uploadsUntil": 0}))
        deleter.reconcile(op["id"])
        assert repo.one(operations, op["id"])["state"] == "completed"
        for prefix in (
            store.key("artifacts", manifest["id"]),
            store.key("staging", allocation["id"]),
        ):
            versions = s3.list_object_versions(Bucket=store.bucket, Prefix=prefix)
            assert not versions.get("Versions") and not versions.get("DeleteMarkers")

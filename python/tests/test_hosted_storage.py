import hashlib
import uuid
from datetime import UTC, datetime

import boto3
import pytest
from moto import mock_aws

from qps.contracts import HostedSubmission, WorkerBatch
from qps.hosted.database import Conflict, Fenced, Repository, artifacts, runs
from qps.hosted.objects import ObjectStore
from qps.hosted.settings import Settings


@pytest.fixture
def registry(local_run):
    _, _, spec = local_run
    spec.spec.execution.max_wall_time_seconds = 300
    settings = Settings(
        auth_mode="development",
        dev_token="owner",
        internal_token="i" * 32,
        worker_signing_key="w" * 32,
        dispatch_enabled=False,
    )
    repo = Repository(settings, create=True)
    request = HostedSubmission(
        run=spec, compute_profile="cpu-fixture", idempotency_key="request-0001"
    )
    run = repo.submit(request, {"modelRevision": "fixture", "datasetIdentity": "fixture"})
    return repo, run, request


def allocate(repo, run):
    return repo.reserve(run["id"], "training", {"hourlyRate": 0, "gpuType": "cpu"})


def test_idempotent_submission_atomic_outbox_and_name_collision(registry):
    repo, run, request = registry
    assert repo.submit(request, {})["id"] == run["id"]
    assert repo.claim_outbox()["resource_id"] == run["id"]
    changed = request.model_copy(deep=True)
    changed.run.spec.training.learning_rate *= 2
    with pytest.raises(Conflict, match="Idempotency"):
        repo.submit(changed, {})
    changed.idempotency_key = "another-key"
    with pytest.raises(Conflict, match="name"):
        repo.submit(changed, {})


def test_environment_expiry_blocks_new_compute_without_a_reservation(registry):
    import time

    from qps.hosted.database import allocations

    repo, run, _ = registry
    repo.settings.environment_deadline = time.time() - 1
    with pytest.raises(Conflict, match="expiry"):
        allocate(repo, run)
    assert repo.list(allocations) == []


def test_event_replays_are_idempotent_and_stale_attempt_is_fenced(registry):
    repo, run, _ = registry
    allocation = allocate(repo, run)
    event = {
        "sequence": 1,
        "time": datetime.now(UTC),
        "kind": "train_step",
        "payload": {"loss": 1.0, "global_step": 1},
    }
    batch = WorkerBatch(events=[event])
    assert repo.ingest(allocation, batch) == 1
    assert repo.ingest(allocation, batch) == 1
    assert len(repo.event_page(run["id"])) == 1
    changed = WorkerBatch(events=[{**event, "payload": {"loss": 2}}])
    with pytest.raises(Conflict):
        repo.ingest(allocation, changed)
    repo.project("TrainingRun", run["id"], "interrupted", "training", 1)
    repo.action(run["id"], "resume")
    with pytest.raises(Fenced):
        repo.authenticate_worker(repo.token(allocation["id"]), allocation["id"])
    with pytest.raises(Conflict, match="cleanup"):
        allocate(repo, run)
    repo.observe_allocation(allocation["id"], {"state": "terminated"})
    assert allocate(repo, run)["generation"] == 2


def test_reservations_hold_slots_until_confirmed_termination(registry):
    repo, run, request = registry
    first = allocate(repo, run)
    other = request.model_copy(deep=True)
    other.idempotency_key = "request-0002"
    other.run.metadata.name = "second"
    second_run = repo.submit(other, {})
    second = allocate(repo, second_run)
    other = request.model_copy(deep=True)
    other.idempotency_key = "request-0003"
    other.run.metadata.name = "third"
    third = repo.submit(other, {})
    with pytest.raises(Conflict, match="slots"):
        allocate(repo, third)
    repo.observe_allocation(first["id"], {"desired": "stop"})
    with pytest.raises(Conflict, match="slots"):
        allocate(repo, third)
    repo.observe_allocation(first["id"], {"state": "terminated"})
    assert allocate(repo, third)["id"] != second["id"]


def test_verified_s3_publication_and_corruption_rejection(registry):
    repo, run, _ = registry
    allocation = allocate(repo, run)
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-2")
        s3.create_bucket(
            Bucket=repo.settings.bucket,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
        )
        s3.put_bucket_versioning(
            Bucket=repo.settings.bucket, VersioningConfiguration={"Status": "Enabled"}
        )
        objects = ObjectStore(repo.settings, repo, s3)
        body = b"actual adapter bytes"
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
        upload = objects.start(allocation, manifest)
        with pytest.raises(KeyError):
            repo.one(artifacts, manifest["id"])
        key = objects.key("staging", allocation["id"], upload["uploadId"], "adapter.safetensors")
        s3.put_object(Bucket=objects.bucket, Key=key, Body=b"corruption")
        with pytest.raises(ValueError, match="checksum"):
            objects.commit(allocation, upload["uploadId"])
        with pytest.raises(KeyError):
            repo.one(artifacts, manifest["id"])
        s3.put_object(Bucket=objects.bucket, Key=key, Body=body)
        record = objects.commit(allocation, upload["uploadId"])
        assert record["objects"]["adapter.safetensors"]["versionId"]
        assert objects.start(allocation, manifest)["committed"]
        assert objects.commit(allocation, upload["uploadId"])["id"] == manifest["id"]
        assert objects.download_descriptor(manifest["id"])["manifestSha"] == record["manifest_sha"]
        repo.project("TrainingRun", run["id"], "interrupted", "training", 1)
        resumed = repo.action(run["id"], "resume", "latest")
        assert repo.attempt(run["id"], resumed["attempt"])["checkpoint"] == manifest["id"]


def test_backup_restore_preserves_identifiers(registry):
    repo, run, _ = registry
    backup = repo.export_metadata()
    restored = Repository(repo.settings, create=True)
    restored.restore_metadata(backup)
    assert restored.one(runs, run["id"])["request_sha"] == run["request_sha"]
    with pytest.raises(Conflict):
        restored.restore_metadata(backup)

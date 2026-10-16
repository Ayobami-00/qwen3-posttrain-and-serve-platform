"""Opt-in real PostgreSQL checks. CI supplies a disposable service, never RDS."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, text
from sqlalchemy.engine import make_url

from qps.contracts import HostedSubmission, WorkerBatch
from qps.hosted.database import Conflict, Repository, allocations, outbox, runs
from qps.hosted.settings import Settings
from qps.specs import load_spec

pytestmark = pytest.mark.skipif(
    not os.environ.get("QPS_TEST_POSTGRES_URL"), reason="Disposable PostgreSQL URL not supplied"
)


@pytest.fixture
def postgres(monkeypatch):
    url = os.environ["QPS_TEST_POSTGRES_URL"]
    schema = "qps_test_" + uuid.uuid4().hex
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA " + schema))
    selected = (
        make_url(url)
        .update_query_dict({"options": "-csearch_path=" + schema})
        .render_as_string(hide_password=False)
    )
    monkeypatch.setenv("QPS_DATABASE_URL", selected)
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(config, "head")
    settings = Settings(
        database_url=selected,
        auth_mode="development",
        dev_token="testing",
        internal_token="i" * 32,
        worker_signing_key="w" * 32,
        runpod_budget_usd=0.8,
        dispatch_enabled=False,
    )
    repo = Repository(settings)
    try:
        yield repo, settings
    finally:
        repo.engine.dispose()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA " + schema + " CASCADE"))
        engine.dispose()


def recipe(name, key):
    value = load_spec(Path(__file__).resolve().parents[2] / "examples/local/gsm8k-lora.yaml")
    value.metadata.name = name
    value.spec.execution.max_wall_time_seconds = 2700
    return HostedSubmission(run=value, idempotency_key=key)


def test_real_migrations_idempotency_and_admission_across_replicas(postgres):
    repo, settings = postgres
    submission = recipe("same-request", "same-request-123")

    def submit(_):
        other = Repository(settings)
        try:
            return other.submit(submission, {})["id"]
        finally:
            other.engine.dispose()

    with ThreadPoolExecutor(max_workers=2) as pool:
        identities = list(pool.map(submit, range(2)))
    assert identities[0] == identities[1]
    second = repo.submit(recipe("second", "second-request-123"), {})

    def reserve(identity):
        other = Repository(settings)
        try:
            try:
                return other.reserve(
                    identity, "training", {"hourlyRate": 0.59, "gpuType": "NVIDIA A40"}
                )
            except Conflict:
                return None
        finally:
            other.engine.dispose()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, [identities[0], second["id"]]))
    accepted = [r for r in results if r]
    assert len(accepted) == 1
    assert sum(r["reserved_usd"] for r in repo.list(allocations)) <= 0.8
    event = {"sequence": 1, "time": datetime.now(UTC), "kind": "train_step", "payload": {"loss": 1}}
    repo.ingest(accepted[0], WorkerBatch(events=[event]))
    backup = repo.export_metadata()
    assert len(backup["tables"]["runs"]) == 2
    assert repo.one(runs, identities[0])["status"] == "queued"
    with repo.transaction() as connection:
        connection.execute(delete(runs))
    repo.restore_metadata(backup)
    assert repo.export_metadata() == backup
    repo.ingest(accepted[0], WorkerBatch(events=[event | {"sequence": 2}]))
    assert [v["id"] for v in repo.event_page(accepted[0]["owner_id"])] == [1, 2]
    repo.submit(recipe("after-restore", "after-restore-123"), {})
    assert max(v["id"] for v in repo.list(outbox)) == 3


def test_managed_backup_keeps_one_snapshot_during_concurrent_submission(postgres, monkeypatch):
    import io
    import json

    from sqlalchemy import event

    from qps.hosted.operations import Operations

    repo, settings = postgres
    writer = Repository(settings)
    captured = {}

    class Storage:
        def put_object(self, **kwargs):
            captured["body"] = kwargs["Body"]

        def get_object(self, **kwargs):
            return {"Body": io.BytesIO(captured["body"])}

    operation = Operations.__new__(Operations)
    operation.secret = lambda _: settings.model_dump()
    operation.bucket, operation.s3 = "fixture-reports", Storage()
    monkeypatch.setattr("qps.hosted.database.Repository", lambda _: repo)
    submitted = []

    def commit_between_table_reads(connection, cursor, statement, *args):
        if statement.lstrip().upper().startswith("SELECT") and not submitted:
            submitted.append(True)
            writer.submit(recipe("during-backup", "during-backup-123"), {})

    original_engine = repo.engine
    event.listen(original_engine, "after_cursor_execute", commit_between_table_reads)
    try:
        operation.backup()
        assert submitted and len(writer.list(runs)) == 1
        backup = json.loads(captured["body"])
        assert backup["tables"]["runs"] == []
        assert backup["tables"]["outbox"] == []
    finally:
        event.remove(original_engine, "after_cursor_execute", commit_between_table_reads)
        writer.engine.dispose()


def test_model_migration_deletion_and_restore_keep_a_live_deployment_independent(postgres):
    from test_hosted_lifecycle import published

    from qps.contracts import DeploymentSubmission
    from qps.hosted.database import deployments, models, operations
    from qps.hosted.deletion import Deleter

    repo, _ = postgres
    request = recipe("independent-model", "independent-0001")
    run = repo.submit(request, {})
    _, _, model, _, objects = published((repo, run, request))
    deployment = repo.deployment_submit(
        DeploymentSubmission(
            name="independent-serving",
            model_version_id=model["id"],
            idempotency_key="independent-serving-1",
        )
    )
    backup = repo.export_metadata()
    operation = repo.start_deletion("experiment", run["id"])
    Deleter(repo, objects).reconcile(operation["id"])
    assert repo.one(operations, operation["id"])["state"] == "completed"
    assert repo.one(deployments, deployment["id"])["model_version_id"] == model["id"]
    repo.restore_metadata(backup)
    assert repo.list(runs) == []
    assert repo.one(deployments, deployment["id"])["model_version_id"] == model["id"]
    assert repo.one(models, model["id"])["status"] == "ready"

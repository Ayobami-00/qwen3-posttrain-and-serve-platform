"""Transactional metadata and admission. No model imports and no provider calls."""

import hashlib
import hmac
import threading
import time
import uuid
from contextlib import contextmanager

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    and_,
    create_engine,
    delete,
    exists,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from qps.contracts import TERMINAL, DeploymentSubmission, HostedSubmission
from qps.hosted.lifecycle import Lifecycle
from qps.util import canonical, fingerprint

metadata = MetaData()


def table(name, *columns, **kwargs):
    return Table(name, metadata, *columns, **kwargs)


runs = table(
    "runs",
    Column("id", String, primary_key=True),
    Column("name", String, unique=True),
    Column("idempotency_key", String, unique=True),
    Column("request_sha", String),
    Column("submission", JSON),
    Column("resolved", JSON),
    Column("status", String),
    Column("phase", String),
    Column("attempt", Integer),
    Column("desired", String),
    Column("created", Float),
    Column("updated", Float),
    Column("projection_time", Float),
)
attempts = table(
    "attempts",
    Column("run_id", String, primary_key=True),
    Column("number", Integer, primary_key=True),
    Column("generation", Integer),
    Column("status", String),
    Column("outputs", JSON),
    Column("checkpoint", String),
    Column("sequence", Integer),
    Column("created", Float),
    Column("reason", String),
)
allocations = table(
    "allocations",
    Column("id", String, primary_key=True),
    Column("owner_id", String),
    Column("kind", String),
    Column("attempt", Integer),
    Column("generation", Integer),
    Column("profile", String),
    Column("state", String),
    Column("desired", String),
    Column("pod_id", String, unique=True),
    Column("gpu_type", String),
    Column("gpu_count", Integer, nullable=False, server_default="1"),
    Column("worker_group", String),
    Column("hourly_rate", Float),
    Column("reserved_usd", Float),
    Column("spent_usd", Float),
    Column("created", Float),
    Column("ended", Float),
    Column("deadline", Float),
    Column("lease_until", Float),
    Column("heartbeat", Float),
    Column("phase", String),
    Column("telemetry", JSON),
    Column("result", JSON),
    Column("endpoint", String),
    Column("ready", Boolean),
    Column("manifest_sha", String),
    UniqueConstraint("owner_id", "generation"),
)
events = table(
    "events",
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String, index=True),
    Column("attempt", Integer),
    Column("generation", Integer),
    Column("sequence", Integer),
    Column("time", String),
    Column("kind", String),
    Column("payload", JSON),
    UniqueConstraint("run_id", "attempt", "sequence"),
)
artifacts = table(
    "artifacts",
    Column("id", String, primary_key=True),
    Column("run_id", String, index=True),
    Column("attempt", Integer),
    Column("kind", String),
    Column("manifest_sha", String),
    Column("manifest", JSON),
    Column("objects", JSON),
    Column("size", Float),
    Column("created", Float),
)
uploads = table(
    "uploads",
    Column("id", String, primary_key=True),
    Column("allocation_id", String),
    Column("manifest", JSON),
    Column("manifest_sha", String),
    Column("state", String),
    Column("created", Float),
    Column("expires", Float),
)
datasets = table(
    "datasets",
    Column("id", String, primary_key=True),
    Column("identity", JSON),
    Column("prepared_key", String, unique=True),
    Column("artifact_id", String),
    Column("created", Float),
)
deployments = table(
    "deployments",
    Column("id", String, primary_key=True),
    Column("name", String, unique=True),
    Column("idempotency_key", String, unique=True),
    Column("request_sha", String),
    Column("spec", JSON),
    Column("run_id", String),
    Column("model_version_id", String),
    Column("status", String),
    Column("desired", String),
    Column("generation", Integer),
    Column("created", Float),
    Column("updated", Float),
)
outbox = table(
    "outbox",
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("resource", String),
    Column("resource_id", String),
    Column("generation", Integer),
    Column("action", String),
    Column("delivered", Boolean),
    Column("locked_until", Float),
    Column("created", Float),
)
budgets = table(
    "budgets",
    Column("id", String, primary_key=True),
    Column("total", Float),
    Column("slots", Integer),
)
sessions = table(
    "sessions",
    Column("id", String, primary_key=True),
    Column("encrypted", String),
    Column("csrf", String),
    Column("expires", Float),
)
tracking = table(
    "tracking",
    Column("run_id", String, primary_key=True),
    Column("attempt", Integer, primary_key=True),
    Column("cursor", Integer),
    Column("url", String),
    Column("state", String),
)


class Conflict(ValueError):
    pass


class Fenced(Conflict):
    pass


models = table(
    "models",
    Column("id", String, primary_key=True),
    Column("source_run_id", String, index=True),
    Column("name", String),
    Column("adapter_id", String),
    Column("artifact_id", String, unique=True),
    Column("snapshot", JSON),
    Column("status", String),
    Column("created", Float),
)
artifact_refs = table(
    "artifact_refs",
    Column("owner_id", String, primary_key=True),
    Column("artifact_id", String, primary_key=True),
    Column("kind", String),
)
provenance = table(
    "artifact_provenance",
    Column("id", String, primary_key=True),
    Column("recipe", JSON),
)
operations = table(
    "operations",
    Column("id", String, primary_key=True),
    Column("resource_id", String, unique=True),
    Column("kind", String),
    Column("state", String),
    Column("phase", String),
    Column("plan", JSON),
    Column("error", String),
    Column("created", Float),
    Column("updated", Float),
    Column("locked_until", Float),
)
tombstones = table(
    "tombstones",
    Column("id", String, primary_key=True),
    Column("kind", String),
    Column("created", Float),
)
cost_ledger = table(
    "cost_ledger",
    Column("id", String, primary_key=True),
    Column("usd", Float),
    Column("gpu_seconds", Float),
    Column("gpu_allocations", Integer, nullable=False, server_default="0"),
    Column("created", Float),
)
tracking_reports = table(
    "tracking_reports",
    Column("id", String, primary_key=True),
    Column("url", String),
    Column("embed_url", String),
    Column("attempts", JSON),
    Column("state", String),
)


class Repository(Lifecycle):
    def __init__(self, settings, *, create=False):
        self.settings = settings
        options = {"pool_pre_ping": True}
        if settings.database_url.startswith("sqlite"):
            options.update(connect_args={"check_same_thread": False}, poolclass=StaticPool)
        self.engine = create_engine(settings.database_url, **options)
        self._lock = threading.RLock()
        if create:
            metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            if not connection.execute(select(budgets).where(budgets.c.id == "project")).first():
                connection.execute(
                    insert(budgets).values(
                        id="project",
                        total=settings.runpod_budget_usd,
                        slots=settings.max_gpu_allocations,
                    )
                )
            connection.execute(
                update(budgets)
                .where(budgets.c.id == "project")
                .values(total=settings.runpod_budget_usd, slots=settings.max_gpu_allocations)
            )
            if not connection.execute(
                select(datasets).where(datasets.c.id == "gsm8k-training-v1")
            ).first():
                connection.execute(
                    insert(datasets).values(
                        id="gsm8k-training-v1",
                        identity={
                            "source": "openai/gsm8k",
                            "split": "train",
                            "normalizer": "gsm8k-v1",
                        },
                        prepared_key=None,
                        artifact_id=None,
                        created=time.time(),
                    )
                )

    @contextmanager
    def transaction(self):
        # The in-process lock also makes the SQLite test implementation deterministic.
        with self._lock, self.engine.begin() as connection:
            yield connection

    @contextmanager
    def connection(self):
        # StaticPool's in-memory SQLite connection must not be rolled back by another
        # thread's read context while a transaction is committing.
        with self._lock, self.engine.connect() as connection:
            yield connection

    def one(self, table, identity, connection=None):
        if connection is None:
            with self.connection() as connection:
                return self.one(table, identity, connection)
        row = connection.execute(select(table).where(table.c.id == identity)).mappings().first()
        if not row:
            raise KeyError(identity)
        return dict(row)

    def list(self, table, *, limit=200, **filters):
        query = select(table)
        for key, value in filters.items():
            query = query.where(table.c[key] == value)
        order = table.c.created if "created" in table.c else table.c.id
        with self.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(query.order_by(order.desc()).limit(limit)).mappings()
            ]

    def locked_one(self, table, identity, connection):
        row = (
            connection.execute(select(table).where(table.c.id == identity).with_for_update())
            .mappings()
            .first()
        )
        if not row:
            raise KeyError(identity)
        return dict(row)

    def _outbox(self, connection, resource, identity, generation, action="run"):
        connection.execute(
            insert(outbox).values(
                resource=resource,
                resource_id=identity,
                generation=generation,
                action=action,
                delivered=False,
                locked_until=0,
                created=time.time(),
            )
        )

    def submit(self, request: HostedSubmission, resolved: dict):
        value = request.model_dump(mode="json", by_alias=True)
        sha = fingerprint(value)
        with self.transaction() as connection:
            self.locked_one(budgets, "project", connection)
            previous = (
                connection.execute(
                    select(runs).where(runs.c.idempotency_key == request.idempotency_key)
                )
                .mappings()
                .first()
            )
            if previous:
                if previous["request_sha"] != sha:
                    raise Conflict("Idempotency key was already used for another submission")
                return dict(previous)
            identity = "run-" + uuid.uuid4().hex
            try:
                connection.execute(
                    insert(runs).values(
                        id=identity,
                        name=request.run.metadata.name,
                        idempotency_key=request.idempotency_key,
                        request_sha=sha,
                        submission=value,
                        resolved=resolved,
                        status="queued",
                        phase="submitted",
                        attempt=1,
                        desired="run",
                        created=time.time(),
                        updated=time.time(),
                        projection_time=None,
                    )
                )
            except IntegrityError as exc:
                raise Conflict("Run name is already in use") from exc
            connection.execute(
                insert(attempts).values(
                    run_id=identity,
                    number=1,
                    generation=1,
                    status="queued",
                    outputs={},
                    checkpoint=None,
                    sequence=0,
                    created=time.time(),
                )
            )
            self._outbox(connection, "TrainingRun", identity, 1)
            return self.one(runs, identity, connection)

    def attempt(self, run_id, number, connection=None):
        if connection is None:
            with self.connection() as connection:
                return self.attempt(run_id, number, connection)
        row = (
            connection.execute(
                select(attempts).where(
                    and_(attempts.c.run_id == run_id, attempts.c.number == number)
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            raise KeyError(f"{run_id}/{number}")
        return dict(row)

    def action(self, run_id, action, checkpoint=None):
        with self.transaction() as connection:
            run = self.locked_one(runs, run_id, connection)
            if run["desired"] == "delete":
                raise Fenced("Experiment is deleting")
            if action == "cancel":
                if run["status"] not in TERMINAL:
                    connection.execute(
                        update(runs)
                        .where(runs.c.id == run_id)
                        .values(desired="cancel", updated=time.time())
                    )
                    self._outbox(connection, "TrainingRun", run_id, run["attempt"], "cancel")
                return self.one(runs, run_id, connection)
            if run["status"] not in TERMINAL:
                raise Conflict("Recovery requires an interrupted or terminal run")
            old = self.attempt(run_id, run["attempt"], connection)
            outputs = dict(old["outputs"])
            if action == "retry-evaluation":
                if not all(
                    phase in outputs for phase in ("prepare", "baseline", "training", "export")
                ):
                    raise Conflict("Evaluation retry requires all earlier verified phase outputs")
                outputs.pop("evaluation", None)
                selected = None
            else:
                query = select(artifacts).where(
                    and_(artifacts.c.run_id == run_id, artifacts.c.kind == "training_checkpoint")
                )
                records = list(connection.execute(query).mappings())
                if checkpoint and checkpoint != "latest":
                    records = [r for r in records if r["id"] == checkpoint]
                selected = max(
                    records, key=lambda r: r["manifest"]["metadata"]["global_step"], default=None
                )
                if checkpoint and not selected:
                    raise Conflict("Checkpoint must be a committed artifact from this run")
                selected = selected["id"] if selected else None
                if selected:
                    for phase in ("training", "export", "evaluation"):
                        outputs.pop(phase, None)
            number = run["attempt"] + 1
            connection.execute(
                insert(attempts).values(
                    run_id=run_id,
                    number=number,
                    generation=number,
                    status="queued",
                    outputs=outputs,
                    checkpoint=selected,
                    sequence=0,
                    created=time.time(),
                )
            )
            connection.execute(
                update(runs)
                .where(runs.c.id == run_id)
                .values(
                    attempt=number,
                    status="queued",
                    phase="submitted",
                    desired="run",
                    updated=time.time(),
                )
            )
            self._outbox(connection, "TrainingRun", run_id, number, action)
            return self.one(runs, run_id, connection)

    def token(self, allocation_id):
        return hmac.new(
            self.settings.worker_signing_key.encode(), allocation_id.encode(), hashlib.sha256
        ).hexdigest()

    def assert_live(self, connection, allocation):
        current = (
            connection.execute(
                select(allocations).where(allocations.c.id == allocation["id"]).with_for_update()
            )
            .mappings()
            .one()
        )
        if (
            current["state"] == "terminated"
            or current["desired"] != "run"
            or min(current["deadline"], current["lease_until"]) < time.time()
        ):
            raise Fenced("Worker lease is no longer active")
        if current["kind"] == "training":
            owner = (
                connection.execute(
                    select(runs).where(runs.c.id == current["owner_id"]).with_for_update()
                )
                .mappings()
                .one()
            )
            if owner["attempt"] != current["generation"] or owner["desired"] != "run":
                raise Fenced("Worker generation was superseded")

    def authenticate_worker(self, token, allocation_id, *, allow_completed=False):
        if not hmac.compare_digest(token, self.token(allocation_id)):
            raise Fenced("Invalid worker credential")
        allocation = self.one(allocations, allocation_id)
        if (
            allocation["state"] == "terminated"
            or allocation["deadline"] < time.time()
            or allocation["lease_until"] < time.time()
        ):
            raise Fenced("Allocation expired")
        if allocation["desired"] != "run" and not allow_completed:
            raise Fenced("Allocation is stopping")
        if allocation["kind"] == "training":
            run = self.one(runs, allocation["owner_id"])
            if run["attempt"] != allocation["generation"] or run["desired"] == "delete":
                raise Fenced("Worker generation was superseded")
        return allocation

    def reserve(self, owner_id, kind, quote):
        if (
            self.settings.environment_deadline
            and time.time() >= self.settings.environment_deadline - 120
        ):
            raise Conflict("Environment expiry prevents new compute admission")
        rate = float(quote["hourlyRate"])
        if not 0 <= rate <= self.settings.max_gpu_hourly_usd * 2:
            raise Conflict("GPU quote exceeds the hourly price ceiling")
        with self.transaction() as connection:
            budget = (
                connection.execute(
                    select(budgets).where(budgets.c.id == "project").with_for_update()
                )
                .mappings()
                .one()
            )
            owner = self.one(runs if kind == "training" else deployments, owner_id, connection)
            generation = owner["attempt"] if kind == "training" else owner["generation"]
            existing = (
                connection.execute(
                    select(allocations).where(
                        and_(
                            allocations.c.owner_id == owner_id,
                            allocations.c.generation == generation,
                        )
                    )
                )
                .mappings()
                .first()
            )
            if existing:
                return dict(existing)
            if owner["desired"] != "run" or owner["status"] in TERMINAL:
                raise Conflict("Workload is no longer runnable")
            for queued_table, revision_column in (
                (runs, runs.c.attempt),
                (deployments, deployments.c.generation),
            ):
                older = connection.execute(
                    select(queued_table.c.id).where(
                        and_(
                            queued_table.c.created < owner["created"],
                            queued_table.c.status == "queued",
                            queued_table.c.desired == "run",
                            ~exists(
                                select(allocations.c.id).where(
                                    and_(
                                        allocations.c.owner_id == queued_table.c.id,
                                        allocations.c.generation == revision_column,
                                    )
                                )
                            ),
                        )
                    )
                ).first()
                if older:
                    raise Conflict("An earlier workload is waiting for FIFO admission")
            profile = (
                owner["submission"]["computeProfile"]
                if kind == "training"
                else owner["spec"]["computeProfile"]
            )
            active = list(
                connection.execute(
                    select(allocations).where(allocations.c.state != "terminated")
                ).mappings()
            )
            gpu_count = (
                0 if profile == "cpu-fixture" else 2 if profile == "runpod-qwen3-fsdp2" else 1
            )
            if quote.get("gpuCount", gpu_count) != gpu_count:
                raise Conflict("GPU quote topology differs from the requested profile")
            if rate > self.settings.max_gpu_hourly_usd * gpu_count:
                raise Conflict("GPU quote exceeds the per-GPU hourly price ceiling")
            prior_starts = sum(connection.execute(select(cost_ledger.c.gpu_allocations)).scalars())
            legacy_ids = set(connection.execute(select(cost_ledger.c.id)).scalars())
            prior_starts += sum(
                a.profile != "cpu-fixture"
                for a in connection.execute(select(allocations))
                if a.id not in legacy_ids and a.state == "terminated"
            )
            if (
                gpu_count
                and prior_starts + sum(a["gpu_count"] > 0 for a in active)
                >= self.settings.max_gpu_starts
            ):
                raise Conflict(
                    "Cumulative GPU startup allowance exhausted; AWS transfer forecast must be reviewed"
                )
            if gpu_count and sum(a["gpu_count"] for a in active) + gpu_count > budget["slots"]:
                raise Conflict("GPU allocation slots are occupied")
            if (
                not gpu_count
                and sum(a["profile"] == "cpu-fixture" for a in active)
                >= self.settings.max_cpu_allocations
            ):
                raise Conflict("CPU allocation slots are occupied")
            if any(a["owner_id"] == owner_id for a in active):
                raise Conflict("Previous allocation cleanup must be confirmed before replacement")
            if kind == "training":
                preparation_key = (owner.get("resolved") or {}).get("preparationKey")
                if preparation_key and not self.prepared(owner, connection):
                    for active_allocation in active:
                        if active_allocation["kind"] != "training":
                            continue
                        other = self.one(runs, active_allocation["owner_id"], connection)
                        if (
                            other["created"] < owner["created"]
                            and (other.get("resolved") or {}).get("preparationKey")
                            == preparation_key
                        ):
                            raise Conflict("Waiting for the shared dataset preparation to commit")
                older = connection.execute(
                    select(runs.c.id).where(
                        and_(
                            runs.c.created < owner["created"],
                            runs.c.status == "queued",
                            runs.c.desired == "run",
                            ~exists(
                                select(allocations.c.id).where(
                                    and_(
                                        allocations.c.owner_id == runs.c.id,
                                        allocations.c.generation == runs.c.attempt,
                                    )
                                )
                            ),
                        )
                    )
                ).first()
                if older:
                    raise Conflict("An earlier submission is waiting for admission")
                configured = owner["submission"]["run"]["spec"]["execution"]["maxWallTimeSeconds"]
            else:
                configured = owner["spec"]["maxRuntimeSeconds"]
            previous = list(
                connection.execute(
                    select(allocations).where(allocations.c.owner_id == owner_id)
                ).mappings()
            )
            consumed = sum(max(0, (a["ended"] or time.time()) - a["created"]) for a in previous)
            seconds = min(configured - consumed, self.settings.allocation_seconds)
            if self.settings.environment_deadline:
                seconds = min(seconds, self.settings.environment_deadline - time.time())
            if seconds < 30:
                raise Conflict("Workload exhausted its original runtime budget")
            reservation = 0 if profile == "cpu-fixture" else (rate + 0.02) * seconds / 3600 + 0.05
            committed = self.spending(connection) + sum(a["reserved_usd"] for a in active)
            if committed + reservation > budget["total"]:
                raise Conflict("Remaining Runpod budget cannot cover this allocation and cleanup")
            identity = "alloc-" + uuid.uuid4().hex
            connection.execute(
                insert(allocations).values(
                    id=identity,
                    owner_id=owner_id,
                    kind=kind,
                    attempt=generation,
                    generation=generation,
                    profile=profile,
                    state="reserved",
                    desired="run",
                    gpu_type=quote.get("gpuType", "cpu"),
                    gpu_count=gpu_count,
                    worker_group=identity,
                    hourly_rate=rate,
                    reserved_usd=reservation,
                    spent_usd=0,
                    created=time.time(),
                    deadline=time.time() + seconds,
                    lease_until=time.time() + 600,
                    heartbeat=None,
                    phase="provisioning",
                    telemetry={},
                    result=None,
                    ready=False,
                )
            )
            return self.one(allocations, identity, connection)

    def observe_allocation(self, identity, observation):
        allowed = {"state", "pod_id", "endpoint", "ready", "manifest_sha", "desired"}
        if not observation.keys() <= allowed:
            raise ValueError("Invalid allocation observation")
        with self.transaction() as connection:
            current = self.one(allocations, identity, connection)
            if current["state"] == "terminated":
                return current
            if observation.get("state") == "terminated":
                ended = time.time()
                observation.update(
                    ended=ended,
                    ready=False,
                    spent_usd=0
                    if current["profile"] == "cpu-fixture"
                    else (ended - current["created"]) / 3600 * (current["hourly_rate"] + 0.02),
                )
            if observation.get("state") == "terminated":
                if not connection.execute(
                    select(cost_ledger).where(cost_ledger.c.id == identity)
                ).first():
                    connection.execute(
                        insert(cost_ledger).values(
                            id=identity,
                            usd=observation["spent_usd"],
                            created=time.time(),
                            gpu_seconds=(ended - current["created"]) * current["gpu_count"],
                            gpu_allocations=int(current["profile"] != "cpu-fixture"),
                        )
                    )
            connection.execute(
                update(allocations).where(allocations.c.id == identity).values(**observation)
            )
            return self.one(allocations, identity, connection)

    def heartbeat(self, allocation, phase, elapsed, telemetry):
        with self.transaction() as connection:
            self.assert_live(connection, allocation)
            current = self.one(allocations, allocation["id"], connection)
            lease_until = min(current["deadline"], time.time() + self.settings.lease_seconds)
            connection.execute(
                update(allocations)
                .where(allocations.c.id == current["id"])
                .values(
                    heartbeat=time.time(), lease_until=lease_until, phase=phase, telemetry=telemetry
                )
            )
            owner = self.one(
                runs if current["kind"] == "training" else deployments,
                current["owner_id"],
                connection,
            )
            return {
                "leaseUntil": lease_until,
                "cancel": owner["desired"] != "run" or current["desired"] != "run",
                "deadline": current["deadline"],
            }

    def ingest(self, allocation, batch):
        with self.transaction() as connection:
            self.assert_live(connection, allocation)
            attempt = self.attempt(allocation["owner_id"], allocation["attempt"], connection)
            cursor = attempt["sequence"]
            for event in batch.events:
                value = event.model_dump(mode="json")
                canonical(value)  # Reject NaN and infinity recursively.
                if len(canonical(value)) > 65536:
                    raise ValueError("Event exceeds 64 KiB")
                previous = (
                    connection.execute(
                        select(events).where(
                            and_(
                                events.c.run_id == allocation["owner_id"],
                                events.c.attempt == allocation["attempt"],
                                events.c.sequence == event.sequence,
                            )
                        )
                    )
                    .mappings()
                    .first()
                )
                if previous:
                    if previous["kind"] != event.kind or previous["payload"] != event.payload:
                        raise Conflict("Event sequence was reused with different contents")
                    continue
                if event.sequence <= cursor:
                    raise Conflict("Event sequence must increase")
                connection.execute(
                    insert(events).values(
                        run_id=allocation["owner_id"],
                        attempt=allocation["attempt"],
                        generation=allocation["generation"],
                        sequence=event.sequence,
                        time=event.time.isoformat(),
                        kind=event.kind,
                        payload=event.payload,
                    )
                )
                cursor = event.sequence
            connection.execute(
                update(attempts)
                .where(
                    and_(
                        attempts.c.run_id == allocation["owner_id"],
                        attempts.c.number == allocation["attempt"],
                    )
                )
                .values(sequence=cursor)
            )
            return cursor

    def event_page(self, run_id, after=0, limit=500, *, kinds=None, attempt=None):
        query = select(events).where(and_(events.c.run_id == run_id, events.c.id > after))
        if kinds:
            query = query.where(events.c.kind.in_(kinds))
        if attempt is not None:
            query = query.where(events.c.attempt == attempt)
        with self.connection() as connection:
            return [
                dict(r)
                for r in connection.execute(query.order_by(events.c.id).limit(limit)).mappings()
            ]

    def phase_output(self, allocation, phase, output):
        with self.transaction() as connection:
            self.assert_live(connection, allocation)
            owner = self.one(runs, allocation["owner_id"], connection)
            cached = self.prepared(owner, connection)
            shared = set()
            if cached:
                prepared_output = cached["identity"]["preparedOutput"]
                shared = {prepared_output["parent"], prepared_output["dataset"]}

            def verified(value):
                if isinstance(value, dict):
                    for child in value.values():
                        verified(child)
                elif isinstance(value, list):
                    for child in value:
                        verified(child)
                elif isinstance(value, str) and value.startswith("art-"):
                    if connection.execute(
                        select(tombstones).where(tombstones.c.id == value)
                    ).first():
                        raise Fenced("Artifact is deleting")
                    artifact = self.one(artifacts, value, connection)
                    self.add_reference(connection, allocation["owner_id"], value, "experiment")
                    if artifact["run_id"] != allocation["owner_id"] and not (
                        value in shared and artifact["kind"] in ("base_model", "dataset")
                    ):
                        raise Conflict("Phase output references another run")

            verified(output)
            attempt = self.attempt(allocation["owner_id"], allocation["attempt"], connection)
            outputs = {**attempt["outputs"], phase: output}
            connection.execute(
                update(attempts)
                .where(
                    and_(
                        attempts.c.run_id == allocation["owner_id"],
                        attempts.c.number == allocation["attempt"],
                    )
                )
                .values(outputs=outputs)
            )
            if phase == "prepare":
                run = self.one(runs, allocation["owner_id"], connection)
                prepared_key = run["resolved"].get("preparationKey") or fingerprint(
                    {
                        "recipe": run["resolved"].get("datasetIdentity"),
                        "dataset": run["submission"]["run"]["spec"]["dataset"],
                        "model": run["resolved"].get("modelRevision"),
                        "sequence": run["submission"]["run"]["spec"]["training"][
                            "maxSequenceLength"
                        ],
                    }
                )
                existing = connection.execute(
                    select(datasets).where(datasets.c.prepared_key == prepared_key)
                ).first()
                if not existing:
                    connection.execute(
                        insert(datasets).values(
                            id="data-" + prepared_key[:32],
                            prepared_key=prepared_key,
                            identity={
                                "source": run["submission"]["run"]["spec"]["dataset"]["id"]
                                if run["submission"]["run"]["spec"]["dataset"]["source"]
                                == "huggingface"
                                else "deterministic-arithmetic-fixture",
                                "split": "train",
                                "preparation": prepared_key,
                                "preparedOutput": output,
                                "sourceRunId": allocation["owner_id"],
                            },
                            artifact_id=output["dataset"],
                            created=time.time(),
                        )
                    )
                resolved = {
                    **run["resolved"],
                    "preparedDatasetVersionId": "data-" + prepared_key[:32],
                }
                connection.execute(
                    update(runs).where(runs.c.id == run["id"]).values(resolved=resolved)
                )
            return outputs

    def prepared(self, owner, connection=None):
        key = (owner.get("resolved") or {}).get("preparationKey")
        if not key:
            return None
        if connection is None:
            with self.connection() as connection:
                return self.prepared(owner, connection)
        row = (
            connection.execute(
                select(datasets).where(
                    and_(
                        datasets.c.prepared_key == key, datasets.c.created > time.time() - 7 * 86400
                    )
                )
            )
            .mappings()
            .first()
        )
        if row and "preparedOutput" in row["identity"]:
            ids = __import__("qps.hosted.lifecycle", fromlist=["references"]).references(
                row["identity"]["preparedOutput"]
            )
            if not connection.execute(select(tombstones).where(tombstones.c.id.in_(ids))).first():
                return dict(row)
        return None

    def finish_worker(self, allocation, result):
        if result.get("status") not in TERMINAL:
            raise ValueError("Worker result requires a terminal status")
        with self.transaction() as connection:
            self.assert_live(connection, allocation)
            if allocation["kind"] == "training" and result["status"] == "succeeded":
                current = self.attempt(allocation["owner_id"], allocation["attempt"], connection)
                if not all(
                    p in current["outputs"]
                    for p in ("prepare", "baseline", "training", "export", "evaluation")
                ):
                    raise Conflict("Run success requires every verified phase output")
            connection.execute(
                update(allocations)
                .where(allocations.c.id == allocation["id"])
                .values(result=result, desired="stop")
            )

    def project(self, resource, identity, status, phase, generation):
        if status not in {"queued", "provisioning", "running", "cancelling", *TERMINAL}:
            raise ValueError("Unknown projected status")
        owner_table = runs if resource == "TrainingRun" else deployments
        with self.transaction() as connection:
            owner = self.locked_one(owner_table, identity, connection)
            if owner["desired"] == "delete":
                return
            expected = owner["attempt"] if resource == "TrainingRun" else owner["generation"]
            if generation != expected:
                raise Fenced("Stale controller projection")
            values = {"status": status, "updated": time.time()}
            if resource == "TrainingRun":
                values.update(phase=phase, projection_time=time.time())
            connection.execute(
                update(owner_table).where(owner_table.c.id == identity).values(**values)
            )
            if resource == "TrainingRun":
                connection.execute(
                    update(attempts)
                    .where(and_(attempts.c.run_id == identity, attempts.c.number == generation))
                    .values(status=status)
                )

    def deployment_submit(self, request: DeploymentSubmission):
        value = request.model_dump(mode="json", by_alias=True)
        sha = fingerprint(value)
        with self.transaction() as connection:
            self.locked_one(budgets, "project", connection)
            previous = (
                connection.execute(
                    select(deployments).where(
                        deployments.c.idempotency_key == request.idempotency_key
                    )
                )
                .mappings()
                .first()
            )
            if previous:
                if previous["request_sha"] != sha:
                    raise Conflict("Idempotency key conflict")
                return dict(previous)
            model = self.resolve_model(request.model_version_id, request.artifact_id, connection)
            if model["status"] != "ready":
                raise Conflict("Model version is not available")
            artifact = self.one(artifacts, model["artifact_id"], connection)
            expected_profile = model["snapshot"]["servingProfile"]
            if request.compute_profile != expected_profile:
                raise Conflict("Deployment profile is incompatible with this model")
            if artifact["created"] + 7 * 86400 <= time.time():
                raise Conflict("Model retention expired; prepare a new run")
            value.update(
                artifactId=artifact["id"],
                modelVersionId=model["id"],
                workerImage=(
                    self.settings.cpu_image
                    if request.compute_profile == "cpu-fixture"
                    else self.settings.gpu_image
                )
                or model["snapshot"].get("workerImage", ""),
            )
            identity = "dep-" + uuid.uuid4().hex
            try:
                connection.execute(
                    insert(deployments).values(
                        id=identity,
                        name=request.name,
                        idempotency_key=request.idempotency_key,
                        request_sha=sha,
                        spec=value,
                        run_id=artifact["run_id"],
                        model_version_id=model["id"],
                        status="queued",
                        desired="run",
                        generation=1,
                        created=time.time(),
                        updated=time.time(),
                    )
                )
            except IntegrityError as exc:
                raise Conflict("Deployment name already exists") from exc
            self._outbox(connection, "ModelDeployment", identity, 1)
            return self.one(deployments, identity, connection)

    def deployment_action(self, identity, action):
        with self.transaction() as connection:
            record = self.locked_one(deployments, identity, connection)
            if record["desired"] == "delete":
                raise Fenced("Deployment is deleting")
            generation = record["generation"]
            if action == "restart":
                if record["status"] not in TERMINAL:
                    raise Conflict("Stop deployment before restarting")
                previous = connection.execute(
                    select(allocations).where(allocations.c.owner_id == identity)
                ).mappings()
                now = time.time()
                consumed = sum(max(0, (a["ended"] or now) - a["created"]) for a in previous)
                remaining = record["spec"]["maxRuntimeSeconds"] - consumed
                if self.settings.environment_deadline:
                    remaining = min(remaining, self.settings.environment_deadline - now)
                if remaining < 30:
                    raise Conflict("Deployment exhausted its original runtime budget")
                generation += 1
            connection.execute(
                update(deployments)
                .where(deployments.c.id == identity)
                .values(
                    desired="cancel" if action == "stop" else "run",
                    generation=generation,
                    status="cancelling" if action == "stop" else "queued",
                    spec={
                        **record["spec"],
                        "workerImage": (
                            self.settings.cpu_image
                            if record["spec"]["computeProfile"] == "cpu-fixture"
                            else self.settings.gpu_image
                        )
                        or record["spec"].get("workerImage", ""),
                    }
                    if action == "restart"
                    else record["spec"],
                    updated=time.time(),
                )
            )
            self._outbox(
                connection,
                "ModelDeployment",
                identity,
                generation,
                "cancel" if action == "stop" else "run",
            )
            return self.one(deployments, identity, connection)

    def budget(self):
        values = self.list(allocations, limit=10000)
        return {
            "limitUsd": self.settings.runpod_budget_usd,
            "estimatedSpentUsd": self.spending(),
            "reservedUsd": sum(a["reserved_usd"] for a in values if a["state"] != "terminated"),
            "activeAllocations": sum(a["state"] != "terminated" for a in values),
            "maxAllocations": self.settings.max_gpu_allocations,
            "activeGpuUnits": sum(a["gpu_count"] for a in values if a["state"] != "terminated"),
            "note": "Estimates include disk allowance; provider billing may arrive later.",
        }

    def claim_outbox(self):
        with self.transaction() as connection:
            row = (
                connection.execute(
                    select(outbox)
                    .where(and_(outbox.c.delivered.is_(False), outbox.c.locked_until < time.time()))
                    .order_by(outbox.c.id)
                    .with_for_update(skip_locked=True)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if row:
                connection.execute(
                    update(outbox)
                    .where(outbox.c.id == row["id"])
                    .values(locked_until=time.time() + 30)
                )
                return dict(row)

    def ack_outbox(self, identity):
        with self.transaction() as connection:
            connection.execute(update(outbox).where(outbox.c.id == identity).values(delivered=True))

    def export_metadata(self):
        with self.connection() as connection:
            return {
                "version": 1,
                "tables": {
                    t.name: [dict(r) for r in connection.execute(select(t)).mappings()]
                    for t in metadata.sorted_tables
                    if t not in (sessions,)
                },
            }

    def restore_metadata(self, data, recovery_ledger=None):
        if data.get("version") != 1:
            raise ValueError("Unsupported backup version")
        with self.transaction() as connection:
            if connection.execute(select(runs).limit(1)).first():
                raise Conflict("Restore requires an empty registry")
            # A restore must not resurrect deleted product data or reset spending.
            retained_tombstones = {
                r["id"]: dict(r) for r in connection.execute(select(tombstones)).mappings()
            }
            retained_ledger = {
                r["id"]: dict(r) for r in connection.execute(select(cost_ledger)).mappings()
            }
            for row in data["tables"].get("tombstones", []):
                retained_tombstones.setdefault(row["id"], row)
            for row in (recovery_ledger or {}).get("tombstones", []):
                retained_tombstones.setdefault(row["id"], row)
            removed = set(retained_tombstones)
            live_references = {
                r["artifact_id"]
                for r in data["tables"].get("artifact_refs", [])
                if r["owner_id"] not in removed and r["artifact_id"] not in removed
            }
            for row in data["tables"].get("artifacts", []):
                if row["run_id"] in removed and row["id"] not in live_references:
                    retained_tombstones.setdefault(
                        row["id"], {"id": row["id"], "kind": "artifact", "created": time.time()}
                    )
            removed = set(retained_tombstones)
            removed_allocations = {
                r["id"] for r in data["tables"].get("allocations", []) if r["owner_id"] in removed
            }
            for row in [
                *data["tables"].get("cost_ledger", []),
                *(recovery_ledger or {}).get("cost_ledger", []),
            ]:
                if (
                    row["id"] not in retained_ledger
                    or row["usd"] > retained_ledger[row["id"]]["usd"]
                ):
                    retained_ledger[row["id"]] = row
            for row in data["tables"].get("allocations", []):
                if row["state"] == "terminated":
                    retained_ledger.setdefault(
                        row["id"],
                        {
                            "id": row["id"],
                            "usd": row["spent_usd"],
                            "gpu_allocations": int(row["profile"] != "cpu-fixture"),
                            "gpu_seconds": max(0, (row["ended"] or row["created"]) - row["created"])
                            * row.get("gpu_count", 1),
                            "created": row["ended"] or row["created"],
                        },
                    )
            for t in reversed(metadata.sorted_tables):
                connection.execute(delete(t))
            for t in metadata.sorted_tables:
                rows = data["tables"].get(t.name, [])
                if t is tombstones:
                    rows = list(retained_tombstones.values())
                elif t is cost_ledger:
                    rows = list(retained_ledger.values())
                elif t is not operations:

                    def survives(row):
                        # artifacts retain historical run_id when a surviving model owns them.
                        keys = (
                            ("id", "artifact_id")
                            if t in (artifacts, provenance, datasets, models, deployments)
                            else (
                                "id",
                                "run_id",
                                "owner_id",
                                "resource_id",
                                "artifact_id",
                                "allocation_id",
                            )
                        )
                        return not any(row.get(k) in removed | removed_allocations for k in keys)

                    rows = [r for r in rows if survives(r)]
                if t is allocations:
                    rows = [
                        {
                            **r,
                            "gpu_count": r.get(
                                "gpu_count", 0 if r["profile"] == "cpu-fixture" else 1
                            ),
                            "worker_group": r.get("worker_group") or r["id"],
                        }
                        for r in rows
                    ]
                if rows:
                    connection.execute(insert(t), rows)
            if self.engine.dialect.name == "postgresql":
                from sqlalchemy import text

                for name in ("events", "outbox"):
                    connection.execute(
                        text(
                            f"SELECT setval(pg_get_serial_sequence('{name}','id'), COALESCE((SELECT MAX(id) FROM {name}),0)+1, false)"
                        )
                    )

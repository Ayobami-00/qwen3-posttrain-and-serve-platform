"""Independent model publication and restartable, fenced resource deletion.

Database intent precedes external cleanup. Operation plans contain identities only;
accounting and tombstones outlive the product resources they describe.
"""

import time
import uuid

from sqlalchemy import delete, exists, insert, select, update


def db():
    from qps.hosted import database

    return database


def references(value):
    if isinstance(value, dict):
        return set().union(*(references(v) for v in value.values()), set())
    if isinstance(value, list):
        return set().union(*(references(v) for v in value), set())
    return {value} if isinstance(value, str) and value.startswith("art-") else set()


class Lifecycle:
    def spending(self, connection=None):
        d = db()
        if connection is None:
            with self.connection() as c:
                return self.spending(c)
        ledger = {r.id: r.usd for r in connection.execute(select(d.cost_ledger))}
        # Legacy rows are counted even before the explicit backfill has run.
        for r in connection.execute(select(d.allocations)):
            ledger.setdefault(r.id, r.spent_usd or 0)
        return sum(ledger.values())

    def add_reference(self, connection, owner_id, artifact_id, kind):
        d = db()
        query = (d.artifact_refs.c.owner_id == owner_id) & (
            d.artifact_refs.c.artifact_id == artifact_id
        )
        if not connection.execute(select(d.artifact_refs).where(query)).first():
            connection.execute(
                insert(d.artifact_refs).values(
                    owner_id=owner_id, artifact_id=artifact_id, kind=kind
                )
            )

    def promote_model(self, run_id, objects=None):
        d = db()
        with self.transaction() as c:
            self.locked_one(d.budgets, "project", c)
            run = self.locked_one(d.runs, run_id, c)
            if run["desired"] == "delete":
                raise d.Fenced("Experiment is deleting")
            all_outputs = self.attempt(run_id, run["attempt"], c)["outputs"]
            if not all(
                p in all_outputs
                for p in ("prepare", "baseline", "training", "export", "evaluation")
            ):
                return None
            owned = [
                dict(r)
                for r in c.execute(
                    select(d.artifacts).where(d.artifacts.c.run_id == run_id)
                ).mappings()
            ]
            exported = references(all_outputs["export"])
            trained = references(all_outputs["training"])
            merged = next(
                (r for r in owned if r["kind"] == "merged_model" and r["id"] in exported), None
            )
            adapter = next(
                (r for r in owned if r["kind"] == "lora_adapter" and r["id"] in trained), None
            )
            if not merged or not adapter:
                raise d.Conflict("Model publication requires verified adapter and merged exports")
            previous = (
                c.execute(select(d.models).where(d.models.c.artifact_id == merged["id"]))
                .mappings()
                .first()
            )
            if previous:
                return dict(previous)
            report = next(
                (
                    r
                    for r in owned
                    if r["kind"] == "comparison_report"
                    and r["id"] in references(all_outputs["evaluation"])
                ),
                None,
            )
            if not report or objects is None:
                return None
            full_report = objects.read_json(report["id"])
            evaluation = {
                part: {
                    k: v
                    for k, v in full_report[part].items()
                    if k
                    in (
                        "accuracy",
                        "correct",
                        "count",
                        "parse_failures",
                        "truncations",
                        "artifact_id",
                        "validation_manifest",
                    )
                }
                for part in ("baseline", "candidate")
            }
            evaluation["accuracy_delta"] = full_report["accuracy_delta"]
            recipe = run["resolved"].get("run", run["submission"]["run"])
            import copy

            serving = copy.deepcopy(recipe)
            cpu = run["submission"]["computeProfile"] == "cpu-fixture"
            serving["spec"]["execution"].update(parallelism="single", gpuCount=0 if cpu else 1)
            snapshot = {
                "recipe": recipe,
                "servingRun": serving,
                "evaluation": evaluation,
                "resolved": run["resolved"],
                "sourceRunId": run_id,
                "servingProfile": "cpu-fixture" if cpu else "runpod-qwen3-small",
                "workerImage": run["resolved"].get("workerImage", ""),
                "parent": merged["manifest"].get("metadata", {}).get("parent"),
                "promptProfile": recipe["spec"]["model"]["promptProfile"],
            }
            identity = "model-" + uuid.uuid4().hex
            c.execute(
                insert(d.models).values(
                    id=identity,
                    name=run["name"],
                    source_run_id=run_id,
                    adapter_id=adapter["id"],
                    artifact_id=merged["id"],
                    snapshot=snapshot,
                    status="ready",
                    created=time.time(),
                )
            )
            for artifact_id in {adapter["id"], merged["id"], *references(all_outputs["prepare"])}:
                self.add_reference(c, identity, artifact_id, "model")
            c.execute(
                delete(d.artifact_refs).where(
                    (d.artifact_refs.c.owner_id == run_id)
                    & d.artifact_refs.c.artifact_id.in_([adapter["id"], merged["id"]])
                )
            )
            return self.one(d.models, identity, c)

    def resolve_model(self, identity, artifact_id, connection):
        d = db()
        if identity:
            model = self.locked_one(d.models, identity, connection)
            if artifact_id and model["artifact_id"] != artifact_id:
                raise d.Conflict("Model version and artifact identity disagree")
            return model
        value = (
            connection.execute(
                select(d.models).where(d.models.c.artifact_id == artifact_id).with_for_update()
            )
            .mappings()
            .first()
        )
        if not value:
            raise d.Conflict("Artifact has no published model version")
        return dict(value)

    def artifact_recipe(self, identity):
        d = db()
        try:
            return self.one(d.provenance, identity)["recipe"]
        except KeyError:
            artifact = self.one(d.artifacts, identity)
            return self.one(d.runs, artifact["run_id"])["resolved"]["run"]

    def backfill_lifecycles(self, objects):
        d = db()
        for run in self.list(d.runs, limit=10000):
            if run["desired"] == "delete":
                continue
            with self.transaction() as c:
                for attempt in c.execute(
                    select(d.attempts).where(d.attempts.c.run_id == run["id"])
                ).mappings():
                    for identity in references(attempt["outputs"]) | (
                        {attempt["checkpoint"]} if attempt["checkpoint"] else set()
                    ):
                        model_owned = c.execute(
                            select(d.models).where(
                                (d.models.c.artifact_id == identity)
                                | (d.models.c.adapter_id == identity)
                            )
                        ).first()
                        if not model_owned:
                            self.add_reference(c, run["id"], identity, "experiment")
                for artifact in c.execute(
                    select(d.artifacts).where(d.artifacts.c.run_id == run["id"])
                ).mappings():
                    if not c.execute(
                        select(d.provenance).where(d.provenance.c.id == artifact["id"])
                    ).first():
                        c.execute(
                            insert(d.provenance).values(
                                id=artifact["id"],
                                recipe=run["resolved"].get("run", run["submission"]["run"]),
                            )
                        )
                for allocation in c.execute(
                    select(d.allocations).where(
                        (d.allocations.c.owner_id == run["id"])
                        & (d.allocations.c.state == "terminated")
                    )
                ).mappings():
                    if not c.execute(
                        select(d.cost_ledger).where(d.cost_ledger.c.id == allocation["id"])
                    ).first():
                        c.execute(
                            insert(d.cost_ledger).values(
                                id=allocation["id"],
                                usd=allocation["spent_usd"],
                                gpu_allocations=int(allocation["profile"] != "cpu-fixture"),
                                gpu_seconds=(allocation["ended"] - allocation["created"])
                                * allocation["gpu_count"],
                                created=time.time(),
                            )
                        )
            if run["status"] == "succeeded":
                self.promote_model(run["id"], objects)
        with self.transaction() as c:
            for deployment in c.execute(select(d.deployments)).mappings():
                model = (
                    c.execute(
                        select(d.models).where(
                            d.models.c.artifact_id == deployment["spec"]["artifactId"]
                        )
                    )
                    .mappings()
                    .first()
                )
                if model:
                    c.execute(
                        update(d.deployments)
                        .where(d.deployments.c.id == deployment["id"])
                        .values(
                            model_version_id=model["id"],
                            spec={**deployment["spec"], "modelVersionId": model["id"]},
                        )
                    )

    def deletion_preview(self, kind, identity, connection=None):
        d = db()
        if connection is None:
            with self.connection() as c:
                return self.deletion_preview(kind, identity, c)
        table = {"experiment": d.runs, "model": d.models, "deployment": d.deployments}[kind]
        owner = self.one(table, identity, connection)
        dependent = (
            [
                dict(r)
                for r in connection.execute(
                    select(d.deployments).where(d.deployments.c.model_version_id == identity)
                ).mappings()
            ]
            if kind == "model"
            else []
        )
        if kind == "deployment" and owner["status"] not in d.TERMINAL:
            raise d.Conflict("Stop the deployment before deleting it")
        owner_ids = [identity, *(r["id"] for r in dependent)]
        owned_artifacts = set(
            connection.execute(
                select(d.artifacts.c.id).where(d.artifacts.c.run_id == identity)
            ).scalars()
        )
        owned_artifacts |= set(
            connection.execute(
                select(d.artifact_refs.c.artifact_id).where(d.artifact_refs.c.owner_id == identity)
            ).scalars()
        )
        retained = set(
            connection.execute(
                select(d.artifact_refs.c.artifact_id).where(
                    ~d.artifact_refs.c.owner_id.in_(owner_ids)
                )
            ).scalars()
        )
        return {
            "resourceId": identity,
            "kind": kind,
            "owners": owner_ids,
            "deployments": [r["id"] for r in dependent],
            "artifacts": sorted(owned_artifacts - retained),
            "preservedArtifacts": sorted(owned_artifacts & retained),
            "allocations": list(
                connection.execute(
                    select(d.allocations.c.id).where(d.allocations.c.owner_id.in_(owner_ids))
                ).scalars()
            ),
            "trackingAttempts": [
                f"{identity}-a{a.number}"
                for a in connection.execute(
                    select(d.attempts).where(d.attempts.c.run_id == identity)
                )
            ]
            if kind == "experiment"
            else [],
            "reportUrl": (
                connection.execute(
                    select(d.tracking_reports.c.url).where(d.tracking_reports.c.id == identity)
                ).scalar()
                if kind == "experiment"
                else None
            ),
            "backupNotice": "Historical backups and infrastructure logs retain their documented expiry.",
        }

    def start_deletion(self, kind, identity):
        d = db()
        with self.transaction() as c:
            self.locked_one(d.budgets, "project", c)
            old = (
                c.execute(select(d.operations).where(d.operations.c.resource_id == identity))
                .mappings()
                .first()
            )
            if old:
                return dict(old)
            plan = self.deletion_preview(kind, identity, c)
            for owner in plan["owners"]:
                table = (
                    d.models
                    if owner.startswith("model-")
                    else d.runs
                    if owner.startswith("run-")
                    else d.deployments
                )
                values = (
                    {"status": "deleting"}
                    if table is d.models
                    else {"status": "deleting", "desired": "delete", "updated": time.time()}
                )
                c.execute(update(table).where(table.c.id == owner).values(**values))
                if table is not d.models:
                    row = self.one(table, owner, c)
                    self._outbox(
                        c,
                        "TrainingRun" if table is d.runs else "ModelDeployment",
                        owner,
                        row.get("attempt", row.get("generation")),
                        "cancel",
                    )
                c.execute(insert(d.tombstones).values(id=owner, kind=kind, created=time.time()))
            c.execute(
                update(d.allocations)
                .where(d.allocations.c.owner_id.in_(plan["owners"]))
                .values(desired="stop")
            )
            # All previously issued PUTs must expire before the final object sweep.
            expiry = (
                c.execute(
                    select(d.uploads.c.expires).where(
                        d.uploads.c.allocation_id.in_(plan["allocations"])
                    )
                )
                .scalars()
                .all()
            )
            plan["uploadsUntil"] = max(expiry, default=0)
            for artifact_id in plan["artifacts"]:
                if not c.execute(
                    select(d.tombstones).where(d.tombstones.c.id == artifact_id)
                ).first():
                    c.execute(
                        insert(d.tombstones).values(
                            id=artifact_id, kind="artifact", created=time.time()
                        )
                    )
            op_id = "op-" + uuid.uuid4().hex
            c.execute(
                insert(d.operations).values(
                    id=op_id,
                    resource_id=identity,
                    kind=kind,
                    state="pending",
                    phase="stopping",
                    plan=plan,
                    created=time.time(),
                    updated=time.time(),
                    locked_until=0,
                )
            )
            return self.one(d.operations, op_id, c)

    def finish_deletion(self, operation):
        d = db()
        plan = operation["plan"]
        with self.transaction() as c:
            self.locked_one(d.budgets, "project", c)
            for artifact_id in plan["artifacts"]:
                if c.execute(
                    select(d.artifact_refs).where(
                        (d.artifact_refs.c.artifact_id == artifact_id)
                        & ~d.artifact_refs.c.owner_id.in_(plan["owners"])
                    )
                ).first():
                    raise d.Conflict("Artifact acquired an unexpected surviving reference")
                if not c.execute(
                    select(d.tombstones).where(d.tombstones.c.id == artifact_id)
                ).first():
                    c.execute(
                        insert(d.tombstones).values(
                            id=artifact_id, kind="artifact", created=time.time()
                        )
                    )
                c.execute(delete(d.artifacts).where(d.artifacts.c.id == artifact_id))
                c.execute(delete(d.provenance).where(d.provenance.c.id == artifact_id))
                c.execute(delete(d.datasets).where(d.datasets.c.artifact_id == artifact_id))
            for owner in plan["owners"]:
                for table, column in (
                    (d.runs, d.runs.c.id),
                    (d.models, d.models.c.id),
                    (d.deployments, d.deployments.c.id),
                    (d.attempts, d.attempts.c.run_id),
                    (d.events, d.events.c.run_id),
                    (d.tracking, d.tracking.c.run_id),
                    (d.tracking_reports, d.tracking_reports.c.id),
                    (d.artifact_refs, d.artifact_refs.c.owner_id),
                    (d.outbox, d.outbox.c.resource_id),
                ):
                    c.execute(delete(table).where(column == owner))
            for allocation_id in plan["allocations"]:
                allocation = self.one(d.allocations, allocation_id, c)
                if allocation["state"] != "terminated":
                    raise d.Conflict("Provider cleanup is not confirmed")
                if not c.execute(
                    select(d.cost_ledger).where(d.cost_ledger.c.id == allocation_id)
                ).first():
                    c.execute(
                        insert(d.cost_ledger).values(
                            id=allocation_id,
                            usd=allocation["spent_usd"],
                            gpu_allocations=int(allocation["profile"] != "cpu-fixture"),
                            gpu_seconds=(allocation["ended"] - allocation["created"])
                            * allocation["gpu_count"],
                            created=time.time(),
                        )
                    )
                c.execute(delete(d.uploads).where(d.uploads.c.allocation_id == allocation_id))
                c.execute(delete(d.allocations).where(d.allocations.c.id == allocation_id))
            # Concurrent deletions may each initially preserve the other's shared inputs.
            # Claim the newly unreferenced inputs transactionally, then purge them in a child
            # operation before reporting the initiating deletion complete.
            orphaned = list(
                c.execute(
                    select(d.artifacts.c.id).where(
                        d.artifacts.c.kind.in_(["base_model", "dataset"])
                        & ~exists(
                            select(d.artifact_refs).where(
                                d.artifact_refs.c.artifact_id == d.artifacts.c.id
                            )
                        )
                        & ~exists(select(d.tombstones).where(d.tombstones.c.id == d.artifacts.c.id))
                    )
                ).scalars()
            )
            if orphaned:
                child = "op-" + uuid.uuid4().hex
                resource = "gc-" + uuid.uuid4().hex
                gc_plan = {
                    "resourceId": resource,
                    "kind": "garbage-collection",
                    "owners": [],
                    "artifacts": orphaned,
                    "preservedArtifacts": [],
                    "allocations": [],
                    "deployments": [],
                    "trackingAttempts": [],
                    "reportUrl": None,
                    "uploadsUntil": 0,
                }
                for identity in orphaned:
                    c.execute(
                        insert(d.tombstones).values(
                            id=identity, kind="artifact", created=time.time()
                        )
                    )
                c.execute(
                    insert(d.operations).values(
                        id=child,
                        resource_id=resource,
                        kind="garbage-collection",
                        state="pending",
                        phase="purging-objects",
                        plan=gc_plan,
                        created=time.time(),
                        updated=time.time(),
                        locked_until=0,
                    )
                )
                c.execute(
                    update(d.operations)
                    .where(d.operations.c.id == operation["id"])
                    .values(
                        state="running",
                        phase="collecting-shared-inputs",
                        plan={
                            "resourceId": plan["resourceId"],
                            "kind": plan["kind"],
                            "childOperationId": child,
                        },
                        updated=time.time(),
                    )
                )
                return
            c.execute(
                update(d.operations)
                .where(d.operations.c.id == operation["id"])
                .values(
                    state="completed",
                    phase="completed",
                    error=None,
                    plan={"resourceId": plan["resourceId"], "kind": plan["kind"]},
                    updated=time.time(),
                )
            )

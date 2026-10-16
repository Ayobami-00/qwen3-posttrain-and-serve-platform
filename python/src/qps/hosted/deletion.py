"""Reconcile durable deletion intent; each external action is safe to replay."""

import logging
import time

from sqlalchemy import select, update

from qps.hosted.database import allocations, operations, outbox, uploads


class Deleter:
    def __init__(self, repository, objects, kubernetes=None, tracking_cleanup=None):
        self.repo, self.objects = repository, objects
        self.kubernetes, self.tracking_cleanup = kubernetes, tracking_cleanup

    def reconcile(self, identity):
        with self.repo.transaction() as c:
            op = self.repo.locked_one(operations, identity, c)
            if op["state"] == "completed" or op["locked_until"] > time.time():
                return
            c.execute(
                update(operations)
                .where(operations.c.id == identity)
                .values(locked_until=time.time() + 120, state="running", error=None)
            )
        try:
            plan = op["plan"]
            records = [self.repo.one(allocations, a) for a in plan["allocations"]]
            if any(a["state"] != "terminated" for a in records):
                self.progress(identity, "stopping")
                return
            if self.kubernetes:
                with self.repo.connection() as c:
                    pending = c.execute(
                        select(outbox).where(
                            outbox.c.resource_id.in_(plan["owners"]) & outbox.c.delivered.is_(False)
                        )
                    ).first()
                if pending:
                    self.progress(identity, "waiting-for-dispatch")
                    return
                for owner in plan["owners"]:
                    if owner.startswith("model-"):
                        continue
                    self.kubernetes.remove(
                        "trainingruns" if owner.startswith("run-") else "modeldeployments", owner
                    )
                for allocation in records:
                    self.kubernetes.remove(
                        "computeallocations",
                        allocation["id"],
                    )
            self.objects.deletion_tombstones(plan)
            for allocation in records:
                self.objects.ledger(allocation)
            self.progress(identity, "tracking")
            if self.tracking_cleanup and (plan["trackingAttempts"] or plan["reportUrl"]):
                self.tracking_cleanup(plan)
            with self.repo.connection() as c:
                expiries = list(
                    c.execute(
                        select(uploads.c.expires).where(
                            uploads.c.allocation_id.in_(plan["allocations"])
                        )
                    ).scalars()
                )
            if max([plan["uploadsUntil"], *expiries]) > time.time():
                self.progress(identity, "waiting-for-upload-expiry")
                return
            self.progress(identity, "purging-objects")
            for artifact in plan["artifacts"]:
                self.objects.purge_prefix(self.objects.key("artifacts", artifact) + "/")
            for allocation in plan["allocations"]:
                self.objects.purge_prefix(self.objects.key("staging", allocation) + "/")
                self.objects.purge_prefix(
                    self.objects.key("allocations", allocation + ".json"), exact=True
                )
            self.repo.finish_deletion(op)
        except Exception:
            # Never persist provider errors containing secrets, signed URLs or response bodies.
            with self.repo.transaction() as c:
                c.execute(
                    update(operations)
                    .where(operations.c.id == identity)
                    .values(
                        state="retrying",
                        error="Cleanup has not completed; the operation will retry.",
                        updated=time.time(),
                    )
                )
            logging.getLogger(__name__).warning("Deletion %s deferred", identity)
        finally:
            with self.repo.transaction() as c:
                c.execute(
                    update(operations).where(operations.c.id == identity).values(locked_until=0)
                )

    def progress(self, identity, phase):
        with self.repo.transaction() as c:
            c.execute(
                update(operations)
                .where(operations.c.id == identity)
                .values(phase=phase, updated=time.time())
            )


def reconcile_deletions(repository, objects, stop):
    from qps.hosted.forge import delete_experiment
    from qps.hosted.kubernetes import Kubernetes

    deleter = Deleter(
        repository,
        objects,
        Kubernetes(repository.settings.namespace) if repository.settings.dispatch_enabled else None,
        delete_experiment if repository.settings.forge_enabled else None,
    )
    while not stop.wait(2):
        with repository.connection() as c:
            identities = list(
                c.execute(
                    select(operations.c.id).where(operations.c.state != "completed")
                ).scalars()
            )
        for identity in identities:
            if stop.is_set():
                return
            deleter.reconcile(identity)

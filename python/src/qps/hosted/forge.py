"""Forge is a projection of durable events, never the artifact source of truth."""

import logging
import os
import threading

import httpx
from sqlalchemy import insert, select, update

from qps.contracts import TERMINAL, portable_recipe
from qps.hosted.database import artifacts, attempts, runs, tracking, tracking_reports
from qps.specs import RunSpec

_projection_lock = threading.RLock()
_handles = {}


def browser_report_url(value):
    """The SDK's app.forge host differs from Forge's public frontend router."""
    if value and value.startswith("https://app.forge.coreweave.com/"):
        return value.replace(
            "https://app.forge.coreweave.com/", "https://forge.coreweave.com/wandb/", 1
        )
    return value


def sync_experiment_report(repository, owner):
    import wandb_workspaces.reports.v2 as wr

    settings = repository.settings
    attempts_ids = [
        f"{owner['id']}-a{a['number']}" for a in repository.list(attempts, run_id=owner["id"])
    ]
    try:
        old = repository.one(tracking_reports, owner["id"])
    except KeyError:
        old = None
    if old and old["attempts"] == sorted(attempts_ids):
        return
    report = (
        wr.Report.from_url(old["url"])
        if old
        else wr.Report(
            entity=settings.forge_entity,
            project=settings.forge_project,
            title=owner["id"],
            description=owner["name"],
        )
    )
    expression = " or ".join(
        f"Config('qps_attempt_id') = '{identity}'" for identity in sorted(attempts_ids)
    )
    report.blocks = [
        wr.P(text=owner["name"] + " — attempts are grouped across infrastructure recovery."),
        wr.PanelGrid(
            runsets=[
                wr.Runset(
                    entity=settings.forge_entity, project=settings.forge_project, filters=expression
                )
            ],
            panels=[
                wr.LinePlot(title="Training loss", x="train/global_step", y=["train/loss"]),
                wr.LinePlot(
                    title="Learning rate", x="train/global_step", y=["train/learning_rate"]
                ),
                wr.LinePlot(title="Gradient norm", x="train/global_step", y=["train/grad_norm"]),
            ],
        ),
    ]
    report.save()
    with repository.transaction() as c:
        current = repository.one(runs, owner["id"], c)
        if current["desired"] == "delete":
            report.delete()
            return
        values = {
            "url": report.url,
            "embed_url": report.url + "?jupyter=true",
            "attempts": sorted(attempts_ids),
            "state": "synced",
        }
        if old:
            c.execute(
                update(tracking_reports)
                .where(tracking_reports.c.id == owner["id"])
                .values(**values)
            )
        else:
            c.execute(insert(tracking_reports).values(id=owner["id"], **values))


def delete_experiment(plan):
    import wandb
    import wandb_workspaces.reports.v2 as wr

    from qps.hosted.settings import Settings

    settings = Settings.from_env()
    with _projection_lock:
        for key in list(_handles):
            if key[0] == plan["resourceId"]:
                _handles.pop(key).finish(exit_code=1)
        api = wandb.Api()
        path = f"{settings.forge_entity}/{settings.forge_project}"
        # Query exact, platform-generated attempt IDs; never delete unrelated project runs.
        filters = {"name": {"$in": plan["trackingAttempts"]}}
        for run in api.runs(path=path, filters=filters):
            if run.id not in plan["trackingAttempts"]:
                raise RuntimeError("Forge run ownership mismatch")
            run.delete(delete_artifacts=False)
        if list(api.runs(path=path, filters=filters)):
            raise RuntimeError("Forge run deletion has not propagated")
        # A report may finish creation after the deletion preview was captured.
        # Discover every exact owned title, including an ambiguous create response.
        for existing in api.reports(path=path):
            if existing.display_name == plan["resourceId"]:
                if not wr.Report.from_url(existing.url).delete():
                    raise RuntimeError("Forge report deletion was not acknowledged")


def setup_project(
    entity, project, key, base_url="https://api.forge.coreweave.com", report_url=None
):
    query = "mutation QPSProject($input: UpsertModelInput!) { upsertModel(input: $input) { project { name access } } }"
    with httpx.Client(timeout=30, trust_env=False, follow_redirects=False) as client:
        policy = client.post(
            base_url + "/graphql",
            auth=("api", key),
            json={
                "query": "query QPSPolicy($name: String!) { entity(name: $name) { privateOnly defaultAccess } }",
                "variables": {"name": entity},
            },
        )
        if policy.status_code != 200 or policy.json().get("errors"):
            raise RuntimeError("Forge team privacy policy could not be verified")
        private_only = policy.json()["data"]["entity"]["privateOnly"]
        response = client.post(
            base_url + "/graphql",
            auth=("api", key),
            json={
                "query": query,
                "variables": {
                    "input": {
                        "name": project,
                        "entityName": entity,
                        **({} if private_only else {"access": "USER_READ"}),
                        "description": "QPS reproducible training and serving systems validation",
                    }
                },
            },
        )
        if response.status_code != 200 or response.json().get("errors"):
            raise RuntimeError("Forge project setup failed; check project creation permissions")
        public = response.json()["data"]["upsertModel"]["project"]["access"] in {
            "PUBLIC",
            "USER_READ",
        }
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_BASE_URL"] = base_url
    import wandb_workspaces.reports.v2 as wr

    if not report_url:
        import wandb

        existing = next(
            (
                r
                for r in wandb.Api().reports(path=f"{entity}/{project}", per_page=10)
                if r.display_name == "QPS experiments"
            ),
            None,
        )
        if existing:
            report_url = existing.url

    if report_url:
        report = wr.Report.from_url(report_url)
    else:
        report = wr.Report(
            project=project,
            entity=entity,
            title="QPS experiments",
            description="Loss, learning rate and validation results. Models and checkpoints are stored privately in S3.",
        )
        report.blocks = [
            wr.P(
                text="Each platform run groups its attempts. Training loss measures the optimization objective; validation accuracy is reported separately."
            ),
            wr.PanelGrid(
                runsets=[wr.Runset(entity=entity, project=project)],
                panels=[
                    wr.LinePlot(title="Training loss", x="train/global_step", y=["train/loss"]),
                    wr.LinePlot(
                        title="Learning rate", x="train/global_step", y=["train/learning_rate"]
                    ),
                    wr.LinePlot(
                        title="Gradient norm", x="train/global_step", y=["train/grad_norm"]
                    ),
                ],
            ),
        ]
        report.save()
    return {
        "entity": entity,
        "project": project,
        "reportUrl": report.url,
        "embedUrl": report.url + "?jupyter=true",
        "public": public,
        "visibilityReason": "Team privateOnly policy" if private_only else "Public project",
    }


def bridge(repository, stop):
    import wandb

    settings = repository.settings
    os.environ["WANDB_BASE_URL"] = settings.forge_base_url
    handles = _handles
    cursors = {}
    closed = set()
    with repository.connection() as connection:
        for record in connection.execute(select(tracking)).mappings():
            key = (record["run_id"], record["attempt"])
            cursors[key] = record["cursor"]
            if record["state"] == "synced":
                closed.add(key)
    os.makedirs(os.environ.get("QPS_TRACKING_DIR", "/tmp/qps-tracking"), exist_ok=True)
    try:
        while not stop.wait(2):
            try:
                with repository.connection() as connection:
                    records = [dict(r) for r in connection.execute(select(attempts)).mappings()]
                with _projection_lock:
                    for attempt in records:
                        key = (attempt["run_id"], attempt["number"])
                        if key in closed:
                            continue
                        page = [
                            e
                            for e in repository.event_page(
                                attempt["run_id"],
                                cursors.get(key, 0),
                                1000,
                                attempt=attempt["number"],
                            )
                        ]
                        owner = repository.one(runs, attempt["run_id"])
                        if owner["desired"] == "delete":
                            if key in handles:
                                handles.pop(key).finish(exit_code=1)
                            continue
                        sync_experiment_report(repository, owner)
                        if key not in handles:
                            handles[key] = wandb.init(
                                entity=settings.forge_entity,
                                project=settings.forge_project,
                                id=attempt["run_id"] + "-a" + str(attempt["number"]),
                                group=attempt["run_id"],
                                name=owner["name"] + " / attempt " + str(attempt["number"]),
                                resume="allow",
                                reinit="create_new",
                                config={
                                    **portable_recipe(
                                        RunSpec.model_validate(owner["resolved"]["run"])
                                    ),
                                    "qps_attempt_id": attempt["run_id"]
                                    + "-a"
                                    + str(attempt["number"]),
                                },
                                settings=wandb.Settings(
                                    base_url=settings.forge_base_url,
                                    disable_git=True,
                                    disable_code=True,
                                    quiet=True,
                                ),
                                dir=os.environ.get("QPS_TRACKING_DIR", "/tmp/qps-tracking"),
                            )
                            handles[key].define_metric("train/global_step")
                            handles[key].define_metric("train/*", step_metric="train/global_step")
                        handle = handles[key]
                        for event in page:
                            if event["kind"] == "train_step":
                                handle.log(
                                    {
                                        **{
                                            "train/" + k: v
                                            for k, v in event["payload"].items()
                                            if v is not None
                                        },
                                        "qps/event_id": event["id"],
                                        "qps/attempt": attempt["number"],
                                    },
                                    step=event["id"],
                                )
                            elif event["kind"] == "comparison_ready":
                                handle.summary.update(event["payload"])
                            cursors[key] = event["id"]
                        if (
                            owner["status"] in TERMINAL or attempt["number"] < owner["attempt"]
                        ) and attempt["number"] <= owner["attempt"]:
                            handle.summary.update(
                                {"qps/status": attempt["status"], "qps/run_id": owner["id"]}
                            )
                            registered = repository.list(artifacts, run_id=owner["id"])
                            handle.summary["qps/artifact_ids"] = {
                                r["id"]: r["kind"] for r in registered
                            }
                            handle.summary["qps/model_revision"] = owner["resolved"].get(
                                "modelRevision"
                            )
                            handle.summary["qps/dataset_revision"] = owner["resolved"].get(
                                "datasetRevision"
                            )
                            comparisons = [
                                r for r in registered if r["kind"] == "comparison_report"
                            ]
                            if comparisons:
                                from qps.hosted.objects import ObjectStore

                                report = ObjectStore(settings, repository).read_json(
                                    comparisons[0]["id"]
                                )
                                handle.summary.update(
                                    {
                                        f"validation/{part}/{field}": report[part][field]
                                        for part in ("baseline", "candidate")
                                        for field in (
                                            "accuracy",
                                            "correct",
                                            "count",
                                            "parse_failures",
                                            "truncations",
                                        )
                                    }
                                )
                            handle.finish(exit_code=0 if attempt["status"] == "succeeded" else 1)
                            del handles[key]
                            closed.add(key)
                            with repository.transaction() as connection:
                                previous = connection.execute(
                                    select(tracking).where(
                                        (tracking.c.run_id == key[0])
                                        & (tracking.c.attempt == key[1])
                                    )
                                ).first()
                                values = {
                                    "cursor": cursors.get(key, 0),
                                    "state": "synced",
                                    "url": f"https://forge.coreweave.com/wandb/{settings.forge_entity}/{settings.forge_project}/runs/{key[0]}-a{key[1]}",
                                }
                                if previous:
                                    connection.execute(
                                        update(tracking)
                                        .where(
                                            (tracking.c.run_id == key[0])
                                            & (tracking.c.attempt == key[1])
                                        )
                                        .values(**values)
                                    )
                                else:
                                    connection.execute(
                                        insert(tracking).values(
                                            run_id=key[0], attempt=key[1], **values
                                        )
                                    )
            except Exception:
                logging.getLogger(__name__).warning(
                    "Forge projection delayed; durable platform events retained"
                )
                stop.wait(5)
    finally:
        for handle in handles.values():
            try:
                handle.finish()
            except Exception:
                pass

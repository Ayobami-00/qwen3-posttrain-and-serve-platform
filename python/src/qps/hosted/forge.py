"""Forge is a projection of durable events, never the artifact source of truth."""

import logging
import os

import httpx
from sqlalchemy import insert, select, update

from qps.contracts import TERMINAL, portable_recipe
from qps.hosted.database import artifacts, attempts, runs, tracking
from qps.specs import RunSpec


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
                        **({} if private_only else {"access": "PUBLIC"}),
                        "description": "QPS reproducible training and serving systems validation",
                    }
                },
            },
        )
        if response.status_code != 200 or response.json().get("errors"):
            raise RuntimeError("Forge project setup failed; check project creation permissions")
        public = response.json()["data"]["upsertModel"]["project"]["access"] in {"PUBLIC", "OPEN"}
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_BASE_URL"] = base_url
    import wandb_workspaces.reports.v2 as wr

    if not report_url:
        import wandb

        existing = next(
            iter(
                wandb.Api().reports(path=f"{entity}/{project}", name="QPS experiments", per_page=10)
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
    handles = {}
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
                for attempt in records:
                    key = (attempt["run_id"], attempt["number"])
                    if key in closed:
                        continue
                    page = [
                        e
                        for e in repository.event_page(
                            attempt["run_id"], cursors.get(key, 0), 1000, attempt=attempt["number"]
                        )
                    ]
                    if not page and key not in handles:
                        continue
                    owner = repository.one(runs, attempt["run_id"])
                    if key not in handles:
                        handles[key] = wandb.init(
                            entity=settings.forge_entity,
                            project=settings.forge_project,
                            id=attempt["run_id"] + "-a" + str(attempt["number"]),
                            group=attempt["run_id"],
                            name=owner["name"] + " / attempt " + str(attempt["number"]),
                            resume="allow",
                            reinit="create_new",
                            config=portable_recipe(
                                RunSpec.model_validate(owner["resolved"]["run"])
                            ),
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
                        comparisons = [r for r in registered if r["kind"] == "comparison_report"]
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
                                    (tracking.c.run_id == key[0]) & (tracking.c.attempt == key[1])
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
                                    insert(tracking).values(run_id=key[0], attempt=key[1], **values)
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

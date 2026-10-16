"""User interface. ML imports remain inside the commands that need them."""

import functools
import json
import time
from pathlib import Path

import typer

from qps.specs import Metadata, RunSpec, absolute_spec, load_spec
from qps.store import Store

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)
runs = typer.Typer(no_args_is_help=True)
models = typer.Typer(no_args_is_help=True)
deployments = typer.Typer(no_args_is_help=True)
fixtures = typer.Typer(no_args_is_help=True)
app.add_typer(runs, name="runs")
app.add_typer(models, name="models")
app.add_typer(deployments, name="deploy")
app.add_typer(fixtures, name="fixtures")


def guarded(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except typer.Exit:
            raise
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

    return wrapped


def output(value):
    typer.echo(json.dumps(value, indent=2, default=str))


def store_for(ctx):
    return Store(ctx.obj)


@app.callback()
def main(ctx: typer.Context, state_dir: Path = typer.Option(Path(".qps"), "--state-dir")):
    """Local Qwen3 training, evaluation and serving. Global options precede commands."""
    ctx.obj = state_dir.resolve()


@app.command()
@guarded
def validate(spec: Path = typer.Option(..., "--spec")):
    """Validate a recipe without importing the ML stack or downloading inputs."""
    parsed = load_spec(spec)
    typer.echo(f"Valid {parsed.api_version} run: {parsed.metadata.name}")


@app.command()
@guarded
def schema(destination: Path | None = typer.Option(None, "--output")):
    """Print or save the public JSON Schema."""
    text = json.dumps(RunSpec.model_json_schema(by_alias=True), indent=2) + "\n"
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    else:
        typer.echo(text)


@app.command()
@guarded
def run(
    ctx: typer.Context,
    spec: Path = typer.Option(..., "--spec"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
):
    """Submit a uniquely named run and execute its complete pipeline."""
    from qps.pipeline import background, coordinate

    store = store_for(ctx)
    record = store.create_run(absolute_spec(load_spec(spec)), spec.read_text())
    typer.echo(f"Run: {record['id']}")
    result = coordinate(store, record["id"]) if wait else background(store, record["id"])
    output(result)


@runs.command("list")
@guarded
def list_runs(ctx: typer.Context):
    from qps.pipeline import reconcile

    store = store_for(ctx)
    output(
        [
            {
                key: reconcile(store, record["id"])[key]
                for key in ("id", "name", "status", "phase", "attempt")
            }
            for record in store.runs()
        ]
    )


@runs.command("show")
@guarded
def show_run(ctx: typer.Context, name: str):
    from qps.pipeline import reconcile

    result = reconcile(store_for(ctx), name)
    for key in ("spec", "resolved", "owner", "worker"):
        result[key] = json.loads(result[key]) if result[key] else None
    output(result)


@runs.command("logs")
@guarded
def logs(ctx: typer.Context, name: str, follow: bool = typer.Option(False, "--follow")):
    from qps.pipeline import TERMINAL, format_event, reconcile

    store = store_for(ctx)
    record = store.run(name)
    cursor = 0
    while True:
        for event in store.events(record["id"], cursor):
            typer.echo(format_event(event))
            cursor = event["id"]
        record = reconcile(store, record["id"])
        if not follow or record["status"] in TERMINAL:
            # A final transaction may have arrived between the last read and status check.
            for event in store.events(record["id"], cursor):
                typer.echo(format_event(event))
            break
        time.sleep(0.2)


@runs.command("resume")
@guarded
def resume(
    ctx: typer.Context, name: str, checkpoint: str | None = typer.Option(None, "--checkpoint")
):
    """Continue verified phases; select a registered checkpoint ID or 'latest'."""
    from qps.pipeline import coordinate

    output(coordinate(store_for(ctx), name, action="resume", checkpoint=checkpoint))


@runs.command("retry")
@guarded
def retry(ctx: typer.Context, name: str, phase: str = typer.Option("evaluation", "--phase")):
    from qps.pipeline import coordinate

    if phase != "evaluation":
        raise ValueError(
            "Use resume for pipeline recovery; targeted retry supports evaluation only"
        )
    output(coordinate(store_for(ctx), name, action="retry-evaluation"))


@runs.command("cancel")
@guarded
def cancel(ctx: typer.Context, name: str):
    from qps.pipeline import reconcile

    store = store_for(ctx)
    record = reconcile(store, name)
    if record["status"] == "pending":
        store.update(record["id"], cancel=1, status="cancelled")
    elif record["status"] == "running":
        store.update(record["id"], cancel=1)
    output(
        {
            "run_id": record["id"],
            "cancellation_requested": True,
            "previous_status": record["status"],
        }
    )


@models.command("list")
@guarded
def list_models(ctx: typer.Context, run: str = typer.Option(..., "--run")):
    store = store_for(ctx)
    output(store.artifacts(store.run(run)["id"]))


@deployments.command("create")
@guarded
def create_deployment(
    ctx: typer.Context,
    from_run: str = typer.Option(..., "--from-run"),
    name: str = typer.Option(..., "--name"),
    backend: str = "transformers",
    bind: str = "127.0.0.1",
    port: int = 8000,
    artifact_output: str = typer.Option("merged_model", "--output"),
):
    from qps.deploy import create

    if backend != "transformers" or artifact_output != "merged_model":
        raise ValueError("v0.1 deploys merged_model artifacts through the transformers backend")
    output(create(store_for(ctx), from_run, name, bind, port))


@deployments.command("stop")
@guarded
def stop_deployment(ctx: typer.Context, name: str):
    from qps.deploy import stop

    store = store_for(ctx)
    stop(store, name)
    output(store.deployment(name))


@deployments.command("list")
@guarded
def list_deployments(ctx: typer.Context):
    store = store_for(ctx)
    with store.connect() as db:
        output([dict(row) for row in db.execute("SELECT * FROM deployments ORDER BY created")])


@app.command()
@guarded
def evaluate(
    ctx: typer.Context,
    deployment: str = typer.Option(..., "--deployment"),
    suite: str = "gsm8k-smoke-validation-v1",
):
    from qps.deploy import evaluate as evaluate_http

    if suite != "gsm8k-smoke-validation-v1":
        raise ValueError("Unsupported evaluation suite")
    output(evaluate_http(store_for(ctx), deployment))


@fixtures.command("create")
@guarded
def create_fixtures(directory: Path = Path(".qps/fixture"), name: str = "tiny-cpu"):
    from qps.fixtures import create_fixture

    Metadata(name=name)
    spec = create_fixture(directory.resolve(), name)
    output(spec.model_dump(mode="json", by_alias=True))


@app.command()
@guarded
def smoke(work_dir: Path = Path(".qps/smoke")):
    """Prove real CPU train/export/evaluate/HTTP/stream/restart behavior offline."""
    from qps.smoke import smoke as run_smoke

    output(run_smoke(work_dir))


if __name__ == "__main__":
    app()

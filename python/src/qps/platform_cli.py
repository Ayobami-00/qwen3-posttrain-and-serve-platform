"""Infrastructure command surface; heavyweight SDKs are imported only on invocation."""

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def invoke(ctx, operation, config=None):
    from qps.hosted.install import Installer

    try:
        result = Installer(Path(ctx.obj), config).execute(operation)
        if result is not None:
            import json

            typer.echo(json.dumps(result, indent=2, default=str))
    except Exception as exc:
        typer.echo(f"Platform {operation} failed: {exc}", err=True)
        raise typer.Exit(1) from None


@app.command()
def doctor(ctx: typer.Context, config: Path | None = None):
    invoke(ctx, "doctor", config)


@app.command()
def init(ctx: typer.Context, config: Path = typer.Option(..., "--config")):
    """Bootstrap scoped state, roles and managed build/cleanup jobs."""
    invoke(ctx, "init", config)


@app.command()
def plan(ctx: typer.Context, config: Path | None = None):
    invoke(ctx, "plan", config)


@app.command()
def apply(ctx: typer.Context, config: Path | None = None):
    invoke(ctx, "apply", config)


@app.command()
def status(ctx: typer.Context, config: Path | None = None):
    invoke(ctx, "status", config)


@app.command()
def destroy(ctx: typer.Context, config: Path | None = None):
    invoke(ctx, "destroy", config)


@app.command()
def backup(ctx: typer.Context, config: Path | None = None):
    invoke(ctx, "backup", config)


@app.command()
def restore(ctx: typer.Context, config: Path | None = None):
    invoke(ctx, "restore", config)

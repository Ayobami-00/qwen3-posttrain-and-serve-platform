"""Hosted CLI transport; cloud and ML libraries remain optional."""

import base64
import hashlib
import json
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import typer

from qps.specs import load_spec

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
runs = typer.Typer(no_args_is_help=True)
models = typer.Typer(no_args_is_help=True)
deploy = typer.Typer(no_args_is_help=True)
app.add_typer(runs, name="runs")
app.add_typer(models, name="models")
app.add_typer(deploy, name="deploy")


def config(ctx):
    path = Path(ctx.obj) / "remote.json"
    if not path.exists():
        raise typer.BadParameter("Run qps remote login first")
    return json.loads(path.read_text())


def request(ctx, method, path, **kwargs):
    value = config(ctx)
    with httpx.Client(
        base_url=value["url"],
        headers={"Authorization": "Bearer " + value["token"]},
        timeout=120,
        trust_env=False,
        follow_redirects=False,
    ) as client:
        response = client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise typer.BadParameter(
                f"Platform returned HTTP {response.status_code}: " + response.text[:300]
            )
        return response.json()


def show(value):
    typer.echo(json.dumps(value, indent=2, default=str))


@app.command()
def login(
    ctx: typer.Context,
    url: str = typer.Option(..., "--url"),
    token_file: Path | None = typer.Option(None, "--token-file"),
):
    """Sign in through Cognito PKCE, or load a pre-issued bearer token from a file."""
    url = url.rstrip("/")
    if not url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
        raise typer.BadParameter("Remote platforms require HTTPS")
    if token_file:
        token = token_file.read_text().strip()
    else:
        info = httpx.get(url + "/api/v1/auth/config", timeout=15, trust_env=False).json()
        verifier = secrets.token_urlsafe(64)
        state = secrets.token_urlsafe(32)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        result = {}

        class Callback(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                params = parse_qs(urlparse(self.path).query)
                if params.get("state", [""])[0] != state:
                    self.send_error(400)
                    return
                result.update(code=params.get("code", [""])[0])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"QPS sign-in complete. You can close this tab.")

        redirect = "http://127.0.0.1:8401/callback"
        query = {
            "client_id": info["clientId"],
            "response_type": "code",
            "scope": "openid profile",
            "redirect_uri": redirect,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        with HTTPServer(("127.0.0.1", 8401), Callback) as server:
            server.timeout = 300
            webbrowser.open(info["domain"] + "/oauth2/authorize?" + urlencode(query))
            server.handle_request()
        if not result.get("code"):
            raise typer.BadParameter("Login timed out")
        response = httpx.post(
            info["domain"] + "/oauth2/token",
            data={
                "grant_type": "authorization_code",
                "client_id": info["clientId"],
                "redirect_uri": redirect,
                "code": result["code"],
                "code_verifier": verifier,
            },
            timeout=15,
            trust_env=False,
        )
        response.raise_for_status()
        token = response.json()["access_token"]
    path = Path(ctx.obj) / "remote.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text(json.dumps({"url": url, "token": token}))
    show(request(ctx, "GET", "/api/v1/auth/me"))


@app.command()
def run(
    ctx: typer.Context,
    spec: Path = typer.Option(..., "--spec"),
    profile: str = "runpod-qwen3-small",
):
    parsed = load_spec(spec)
    body = {
        "run": parsed.model_dump(mode="json", by_alias=True),
        "computeProfile": profile,
        "datasetVersionId": "gsm8k-training-v1",
        "idempotencyKey": secrets.token_hex(16),
    }
    show(request(ctx, "POST", "/api/v1/runs", json=body))


@runs.command("list")
def list_runs(ctx: typer.Context):
    show(request(ctx, "GET", "/api/v1/runs"))


@runs.command("show")
def show_run(ctx: typer.Context, identity: str):
    show(request(ctx, "GET", "/api/v1/runs/" + identity))


@runs.command("logs")
def logs(ctx: typer.Context, identity: str, follow: bool = False):
    cursor = 0
    while True:
        page = request(ctx, "GET", f"/api/v1/runs/{identity}/events", params={"after": cursor})
        for event in page:
            cursor = event["id"]
            typer.echo(json.dumps(event))
        if not follow:
            break
        status = request(ctx, "GET", "/api/v1/runs/" + identity)["status"]
        if status in ("succeeded", "failed", "cancelled", "interrupted"):
            break
        time.sleep(1)


def action(ctx, identity, name, checkpoint=None):
    show(
        request(
            ctx,
            "POST",
            f"/api/v1/runs/{identity}/actions",
            json={"action": name, "checkpoint": checkpoint},
        )
    )


@runs.command("cancel")
def cancel(ctx: typer.Context, identity: str):
    action(ctx, identity, "cancel")


@runs.command("resume")
def resume(ctx: typer.Context, identity: str, checkpoint: str = "latest"):
    action(ctx, identity, "resume", checkpoint)


@runs.command("retry")
def retry(ctx: typer.Context, identity: str):
    action(ctx, identity, "retry-evaluation")


@models.command("list")
def model_list(ctx: typer.Context, run: str = typer.Option(..., "--run")):
    show(request(ctx, "GET", "/api/v1/artifacts", params={"run": run}))


@deploy.command("create")
def deployment(ctx: typer.Context, artifact: str, name: str, profile: str = "runpod-qwen3-small"):
    show(
        request(
            ctx,
            "POST",
            "/api/v1/deployments",
            json={
                "artifactId": artifact,
                "name": name,
                "computeProfile": profile,
                "idempotencyKey": secrets.token_hex(16),
            },
        )
    )


@deploy.command("stop")
def stop(ctx: typer.Context, identity: str):
    show(request(ctx, "POST", f"/api/v1/deployments/{identity}/stop"))


@deploy.command("restart")
def restart(ctx: typer.Context, identity: str):
    show(request(ctx, "POST", f"/api/v1/deployments/{identity}/restart"))

"""Hosted application, worker protocol and authenticated inference gateway."""

import asyncio
import hmac
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field
from sqlalchemy import select

from qps.contracts import (
    BackendCapability,
    DeletionOperation,
    DeploymentSubmission,
    Heartbeat,
    HostedSubmission,
    ModelVersion,
    PhaseOutput,
    WorkerBatch,
    recipe_digest,
)
from qps.hosted.auth import Auth
from qps.hosted.database import (
    Conflict,
    Fenced,
    Repository,
    allocations,
    artifacts,
    attempts,
    budgets,
    datasets,
    deployments,
    events,
    models,
    operations,
    runs,
    tracking_reports,
)
from qps.hosted.http import until_disconnect
from qps.hosted.objects import ObjectStore
from qps.hosted.settings import Settings
from qps.specs import Contract
from qps.util import canonical, fingerprint


class RunAction(Contract):
    action: Literal["cancel", "resume", "retry-evaluation"]
    checkpoint: str | None = None


class Quote(Contract):
    hourly_rate: float = Field(ge=0, le=2, allow_inf_nan=False)
    gpu_count: int | None = Field(default=None, ge=0, le=2)
    gpu_type: str


class Projection(Contract):
    status: str
    phase: str = ""
    generation: int = Field(ge=1)


class Result(Contract):
    status: Literal["succeeded", "failed", "cancelled", "interrupted"]
    reason: str = Field(default="", max_length=2000)


def resolve(request):
    spec = request.run.model_copy(deep=True)
    if request.compute_profile != "cpu-fixture":
        from huggingface_hub import HfApi

        api = HfApi(token=False)
        spec.spec.model.revision = api.model_info(
            spec.spec.model.id, revision=spec.spec.model.revision
        ).sha
        spec.spec.dataset.revision = api.dataset_info(
            spec.spec.dataset.id, revision=spec.spec.dataset.revision
        ).sha
        if not spec.spec.model.revision or not spec.spec.dataset.revision:
            raise ValueError("Hub revision could not be resolved")
    result = {
        "run": spec.model_dump(mode="json", by_alias=True),
        "recipeSha": recipe_digest(spec),
        "modelRevision": spec.spec.model.revision,
        "datasetRevision": spec.spec.dataset.revision,
        "datasetIdentity": fingerprint(spec.spec.dataset.model_dump(mode="json", by_alias=True)),
    }
    result["preparationKey"] = fingerprint(
        {
            "model": spec.spec.model.model_dump(mode="json", by_alias=True),
            "dataset": spec.spec.dataset.model_dump(mode="json", by_alias=True),
            "maxSequenceLength": spec.spec.training.max_sequence_length,
            "generationBudget": spec.spec.evaluation.max_new_tokens,
        }
    )
    return result


def public_run(record):
    submitted = record["submission"]["run"]["spec"]
    return {
        key: record[key]
        for key in (
            "id",
            "name",
            "status",
            "phase",
            "attempt",
            "created",
            "updated",
            "projection_time",
        )
    } | {
        "model": submitted["model"]["id"]
        if submitted["model"]["source"] == "huggingface"
        else "tiny-qwen3-fixture",
        "dataset": (record.get("resolved") or {}).get("preparedDatasetVersionId")
        or record["submission"]["datasetVersionId"],
        "training": submitted["training"],
        "parallelism": submitted["execution"].get("parallelism", "single"),
        "gpuCount": submitted["execution"]["gpuCount"],
        "effectiveBatchSize": submitted["training"]["microBatchSize"]
        * submitted["training"]["gradientAccumulationSteps"]
        * max(1, submitted["execution"]["gpuCount"]),
        "evaluation": submitted["evaluation"],
        "forgeGroup": record["id"],
    }


PUBLIC_METRICS = {
    "global_step",
    "loss",
    "learning_rate",
    "grad_norm",
    "elapsed_seconds",
    "step_seconds",
    "tokens_seen",
    "supervised_tokens_seen",
    "cuda_peak_allocated_bytes",
    "correct",
    "count",
    "accuracy",
    "parse_failures",
    "truncations",
    "generation_failures",
    "baseline",
    "candidate",
}


def create_app(settings=None, repository=None, objects=None):
    settings = settings or Settings.from_env()
    repository = repository or Repository(settings)
    objects = objects or ObjectStore(settings, repository)
    auth = Auth(settings, repository)
    stop = threading.Event()

    @asynccontextmanager
    async def lifespan(app):
        repository.backfill_lifecycles(objects)
        threads = []
        from qps.hosted.deletion import reconcile_deletions

        threads.append(
            threading.Thread(
                target=reconcile_deletions, args=(repository, objects, stop), daemon=True
            )
        )
        if settings.dispatch_enabled:
            from qps.hosted.kubernetes import dispatch

            threads.append(threading.Thread(target=dispatch, args=(repository, stop), daemon=True))
        if settings.forge_enabled:
            from qps.hosted.forge import bridge

            threads.append(threading.Thread(target=bridge, args=(repository, stop), daemon=True))
        for thread in threads:
            thread.start()
        yield
        stop.set()
        for thread in threads:
            thread.join(timeout=5)

    app = FastAPI(title="QPS hosted platform", version="0.4.0", lifespan=lifespan)
    app.state.repository = repository
    app.state.objects = objects
    app.state.auth = auth

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Resource not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.middleware("http")
    async def security_headers(request, call_next):
        if (
            request.headers.get("content-length", "0").isdigit()
            and int(request.headers.get("content-length", "0")) > 2 * 1024**2
        ):
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        if request.method in ("POST", "PUT", "PATCH"):
            received = bytearray()
            async for chunk in request.stream():
                received.extend(chunk)
                if len(received) > 2 * 1024**2:
                    return JSONResponse({"detail": "Request body too large"}, status_code=413)
            request._body = bytes(received)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        return response

    def internal(request):
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(token, settings.internal_token):
            raise HTTPException(401, "Controller authentication required")

    def worker(request, identity, allow_completed=False):
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        try:
            return repository.authenticate_worker(token, identity, allow_completed=allow_completed)
        except Fenced as exc:
            raise HTTPException(409, str(exc)) from exc

    def workload(kind, identity):
        return repository.one(runs if kind == "training" else deployments, identity)

    def allocation_view(allocation):
        owner = workload(allocation["kind"], allocation["owner_id"])
        value = dict(allocation)
        value.update(
            workerToken=repository.token(allocation["id"]),
            apiUrl=os.environ.get("QPS_WORKER_API_URL", settings.public_url),
            image=(owner.get("resolved") or {}).get("workerImage")
            or owner.get("spec", {}).get("workerImage")
            or (
                settings.cpu_image if allocation["profile"] == "cpu-fixture" else settings.gpu_image
            ),
            ownerDesired=owner["desired"],
            ownerStatus=owner["status"],
            resourceName="qps-test-" + allocation["id"],
            namespace=settings.namespace,
        )
        if allocation["kind"] == "deployment":
            artifact = repository.one(artifacts, owner["spec"]["artifactId"])
            value.update(
                expectedArtifactId=artifact["id"], expectedManifestSha=artifact["manifest_sha"]
            )
        return value

    @app.get("/healthz")
    def health():
        return {"status": "alive"}

    @app.get("/readyz")
    def ready():
        with repository.connection() as connection:
            connection.execute(select(runs.c.id).limit(1))
        return {"status": "ready", "version": "0.4.0"}

    @app.get("/api/v1/auth/config")
    def auth_config():
        return {
            "mode": settings.auth_mode,
            "clientId": settings.cognito_client_id,
            "domain": settings.cognito_domain,
            "issuer": settings.cognito_issuer,
        }

    @app.get("/api/v1/auth/login")
    def login():
        return auth.login()

    @app.get("/api/v1/auth/callback")
    def callback(request: Request, code: str, state: str):
        return auth.callback(request, code, state)

    @app.get("/api/v1/auth/me")
    def me(request: Request):
        owner = auth.owner(request, required=False)
        return {
            "authenticated": bool(owner),
            "username": "owner" if owner else None,
            "csrfToken": owner.get("csrf") if owner else None,
        }

    @app.post("/api/v1/auth/logout")
    def logout(request: Request):
        auth.owner(request)
        return auth.logout(request)

    @app.post("/api/v1/auth/development")
    def development_login(request: Request):
        if settings.auth_mode != "development":
            raise HTTPException(404, "Not found")
        auth.owner(request)
        response = JSONResponse({"authenticated": True, "csrfToken": settings.dev_token})
        response.set_cookie(
            "qps_dev", settings.dev_token, httponly=True, samesite="strict", max_age=3600
        )
        return response

    @app.get("/api/v1/compute-profiles")
    def profiles():
        return [
            {
                "id": "runpod-qwen3-small",
                "label": "Qwen3 single GPU",
                "gpuCount": 1,
                "minVramGiB": 24,
                "dtype": "bf16",
                "maxRuntimeSeconds": 2700,
                "maxGpuHourlyUsd": settings.max_gpu_hourly_usd,
            },
            {
                "id": "runpod-qwen3-fsdp2",
                "label": "Qwen3 FSDP2 · two GPUs on one host",
                "gpuCount": 2,
                "dtype": "bf16",
                "minVramGiB": 24,
                "maxRuntimeSeconds": 2700,
                "maxGpuHourlyUsd": settings.max_gpu_hourly_usd,
            },
            {
                "id": "cpu-fixture",
                "label": "Offline CPU fixture",
                "gpuCount": 0,
                "dtype": "fp32",
                "maxRuntimeSeconds": 300,
            },
        ]

    @app.get("/api/v1/backends")
    def backends():
        return [BackendCapability().model_dump(by_alias=True)]

    @app.get("/api/v1/datasets")
    def list_datasets():
        return [
            {
                "id": row["id"],
                "artifactId": row["artifact_id"],
                "created": row["created"],
                "identity": {
                    key: value
                    for key, value in row["identity"].items()
                    if key not in ("preparedOutput", "sourceRunId")
                },
            }
            for row in repository.list(datasets)
        ]

    @app.get("/api/v1/runs")
    def list_runs(request: Request, limit: int = Query(100, ge=1, le=200)):
        return [public_run(r) for r in repository.list(runs, limit=limit)]

    @app.post("/api/v1/runs", status_code=201)
    def submit(request: Request, body: HostedSubmission):
        auth.owner(request)
        if repository.list(runs, idempotency_key=body.idempotency_key):
            return public_run(repository.submit(body, {}))
        repository.one(datasets, body.dataset_version_id)
        resolved = resolve(body)
        resolved["workerImage"] = (
            settings.cpu_image if body.compute_profile == "cpu-fixture" else settings.gpu_image
        )
        resolved["preparationKey"] = fingerprint(
            {"inputs": resolved["preparationKey"], "image": resolved["workerImage"]}
        )
        selected_dataset = repository.one(datasets, body.dataset_version_id)
        if (
            selected_dataset["prepared_key"]
            and selected_dataset["prepared_key"] != resolved["preparationKey"]
        ):
            raise ValueError("Selected prepared dataset is incompatible with this recipe/tokenizer")
        return public_run(repository.submit(body, resolved))

    @app.get("/api/v1/runs/{identity}")
    def get_run(identity: str, request: Request):
        run = repository.one(runs, identity)
        result = public_run(run)
        if auth.owner(request, required=False):
            result.update(
                submission=run["submission"],
                resolved=run["resolved"],
                attempts=repository.list(attempts, run_id=identity),
                allocations=repository.list(allocations, owner_id=identity),
                artifacts=repository.list(artifacts, run_id=identity),
            )
        return result

    @app.post("/api/v1/runs/{identity}/actions")
    def run_action(identity: str, request: Request, body: RunAction):
        auth.owner(request)
        return public_run(repository.action(identity, body.action, body.checkpoint))

    @app.get("/api/v1/runs/{identity}/metrics")
    def metrics(identity: str, after: int = Query(0, ge=0)):
        repository.one(runs, identity)
        return [
            {
                **{key: e[key] for key in ("id", "attempt", "time", "kind")},
                "payload": {k: v for k, v in e["payload"].items() if k in PUBLIC_METRICS},
            }
            for e in repository.event_page(
                identity,
                after,
                1000,
                kinds=("train_step", "evaluation_complete", "comparison_ready"),
            )
        ]

    @app.get("/api/v1/runs/{identity}/telemetry")
    def telemetry(identity: str):
        repository.one(runs, identity)
        return [
            {
                "attempt": a["attempt"],
                "workerGroup": a["worker_group"] or a["id"],
                "worker": a["worker_group"] or a["id"],
                "rankHealth": a["telemetry"].get("rank_health", []),
                "phase": a["phase"],
                "lastSeen": a["heartbeat"],
                "stale": a["state"] == "terminated"
                or not a["heartbeat"]
                or time.time() - a["heartbeat"] > 45,
                "gpus": a["telemetry"].get("gpus", []),
            }
            for a in repository.list(allocations, owner_id=identity)
        ]

    @app.get("/api/v1/runs/{identity}/events")
    async def event_feed(
        identity: str, request: Request, after: int = Query(0, ge=0), follow: bool = False
    ):
        auth.owner(request)
        repository.one(runs, identity)
        if not follow:
            return repository.event_page(identity, after)
        cursor = max(after, int(request.headers.get("last-event-id", "0")))

        async def generate():
            nonlocal cursor
            started = time.monotonic()
            while not await request.is_disconnected() and time.monotonic() - started < 240:
                page = repository.event_page(identity, cursor)
                for event in page:
                    cursor = event["id"]
                    yield f"id: {cursor}\ndata: {canonical(event)}\n\n"
                if not page:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    @app.get("/api/v1/artifacts")
    def list_artifacts(run: str | None = None):
        records = repository.list(artifacts, **({"run_id": run} if run else {}))
        return [
            {
                k: r[k]
                for k in ("id", "run_id", "attempt", "kind", "manifest_sha", "size", "created")
            }
            for r in records
        ]

    @app.get("/api/v1/artifacts/{identity}")
    def get_artifact(identity: str, request: Request):
        record = repository.one(artifacts, identity)
        result = {
            k: record[k]
            for k in ("id", "run_id", "attempt", "kind", "manifest_sha", "size", "created")
        }
        if auth.owner(request, required=False):
            result["manifest"] = record["manifest"]
        return result

    @app.get("/api/v1/artifacts/{identity}/download")
    def download(identity: str, request: Request):
        auth.owner(request)
        return objects.download_descriptor(identity)

    @app.get("/api/v1/runs/{identity}/comparison")
    def comparison(identity: str, request: Request):
        records = repository.list(artifacts, run_id=identity, kind="comparison_report")
        if not records:
            raise HTTPException(404, "Comparison is not available yet")
        report = objects.read_json(records[0]["id"])
        if auth.owner(request, required=False):
            return report
        return {
            key: {k: v for k, v in report[key].items() if k in PUBLIC_METRICS}
            for key in ("baseline", "candidate")
        } | {"accuracy_delta": report["accuracy_delta"]}

    @app.api_route(
        "/observability/grafana/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        include_in_schema=False,
    )
    async def grafana(path: str, request: Request):
        if (
            request.method not in {"GET", "HEAD"}
            and request.headers.get("origin") != settings.public_url
        ):
            raise HTTPException(403, "Same-origin request required")
        auth.owner(request, check_csrf=False)
        async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False) as client:
            response = await client.request(
                request.method,
                "http://qps-monitoring-grafana.monitoring.svc/observability/grafana/" + path,
                params=request.query_params,
                content=await request.body(),
                headers={
                    "X-WEBAUTH-USER": "owner",
                    "Content-Type": request.headers.get("content-type", "application/json"),
                },
            )
        return Response(
            response.content,
            status_code=response.status_code,
            headers={
                k: v
                for k, v in response.headers.items()
                if k.lower() in {"content-type", "location", "cache-control"}
            },
        )

    @app.get("/api/v1/tracking")
    def tracking_info(request: Request):
        return {
            "entity": settings.forge_entity,
            "project": settings.forge_project,
            "reportUrl": settings.forge_report_url if settings.forge_report_public else None,
            "reportLink": settings.forge_report_url
            if auth.owner(request, required=False)
            else None,
            "privateOnly": bool(settings.forge_report_url) and not settings.forge_report_public,
            "projectUrl": f"https://forge.coreweave.com/wandb/{settings.forge_entity}/{settings.forge_project}",
        }

    @app.get("/api/v1/deployments")
    def list_deployments(request: Request):
        owner = auth.owner(request, required=False)
        return [
            {k: d[k] for k in ("id", "name", "status", "created", "updated")}
            | {
                "spec": d["spec"]
                if owner
                else {
                    k: v
                    for k, v in d["spec"].items()
                    if k
                    in {
                        "artifactId",
                        "backend",
                        "maxContextTokens",
                        "maxOutputTokens",
                    }
                }
            }
            for d in repository.list(deployments)
        ]

    @app.post("/api/v1/deployments", status_code=201)
    def create_deployment(request: Request, body: DeploymentSubmission):
        auth.owner(request)
        return repository.deployment_submit(body)

    @app.post("/api/v1/deployments/{identity}/{action}")
    def deployment_action(identity: str, action: Literal["stop", "restart"], request: Request):
        auth.owner(request)
        return repository.deployment_action(identity, action)

    @app.get("/api/v1/models")
    def list_models(request: Request):
        owner = auth.owner(request, required=False)
        records = repository.list(models)
        if owner:
            return records
        return [
            {k: r[k] for k in ("id", "name", "artifact_id", "status", "created")} for r in records
        ]

    @app.get(
        "/api/v1/models/{identity}", response_model=ModelVersion, response_model_by_alias=False
    )
    def get_model(identity: str, request: Request):
        auth.owner(request)
        return repository.one(models, identity)

    @app.get("/api/v1/{resource}/{identity}/deletion-preview")
    def preview_deletion(
        resource: Literal["runs", "models", "deployments"], identity: str, request: Request
    ):
        auth.owner(request)
        preview = repository.deletion_preview(
            {"runs": "experiment", "models": "model", "deployments": "deployment"}[resource],
            identity,
        )
        existing = repository.list(operations, resource_id=identity)
        return {**preview, "operation": existing[0] if existing else None}

    @app.delete("/api/v1/{resource}/{identity}", status_code=202)
    def delete_resource(
        resource: Literal["runs", "models", "deployments"],
        identity: str,
        request: Request,
        confirm: str,
    ):
        auth.owner(request)
        if confirm != identity:
            raise HTTPException(422, "Confirmation must match the resource identity")
        return repository.start_deletion(
            {"runs": "experiment", "models": "model", "deployments": "deployment"}[resource],
            identity,
        )

    @app.get(
        "/api/v1/operations/{identity}",
        response_model=DeletionOperation,
        response_model_by_alias=False,
    )
    def get_operation(identity: str, request: Request):
        auth.owner(request)
        return repository.one(operations, identity)

    @app.get("/api/v1/runs/{identity}/tracking")
    def experiment_tracking(identity: str):
        repository.one(runs, identity)
        try:
            report = repository.one(tracking_reports, identity)
        except KeyError:
            return {"state": "pending", "embedUrl": None, "reportUrl": None}
        return {
            "state": report["state"],
            "reportUrl": report["url"],
            "embedUrl": report["embed_url"] if settings.forge_report_public else None,
        }

    @app.get("/api/v1/experiments/comparison")
    def compare_experiments(ids: str):
        identities = ids.split(",")
        if not 2 <= len(identities) <= 8:
            raise HTTPException(422, "Compare between two and eight experiments")
        result = []
        for identity in identities:
            run = repository.one(runs, identity)
            reports = repository.list(artifacts, run_id=identity, kind="comparison_report")
            report = objects.read_json(reports[0]["id"]) if reports else None
            profile = run["resolved"].get("run", run["submission"]["run"])["spec"]["evaluation"]
            prepared = repository.attempt(identity, run["attempt"])["outputs"].get("prepare", {})
            manifest = (
                repository.one(artifacts, prepared["dataset"])["manifest_sha"]
                if prepared.get("dataset")
                else None
            )
            result.append(
                {
                    "runId": identity,
                    "name": run["name"],
                    "status": "complete"
                    if report
                    else "pending"
                    if run["status"] not in {"failed", "cancelled", "interrupted"}
                    else "unavailable",
                    "validationManifest": manifest,
                    "generation": profile,
                    "baseline": {k: v for k, v in report["baseline"].items() if k in PUBLIC_METRICS}
                    if report
                    else None,
                    "candidate": {
                        k: v for k, v in report["candidate"].items() if k in PUBLIC_METRICS
                    }
                    if report
                    else None,
                    "improvementPercentagePoints": 100 * report["accuracy_delta"]
                    if report
                    else None,
                }
            )
        comparable = bool(result[0]["validationManifest"]) and all(
            r["validationManifest"] == result[0]["validationManifest"]
            and r["generation"] == result[0]["generation"]
            for r in result
        )
        return {
            "comparable": comparable,
            "experiments": result,
            "notice": None
            if comparable
            else "Validation manifests or generation settings differ or are unavailable.",
        }

    @app.get("/api/v1/admin/budget")
    def budget(request: Request):
        auth.owner(request)
        return repository.budget()

    @app.get("/api/internal/v1/queue")
    def queue(request: Request):
        internal(request)
        return {
            "training": repository.list(runs, status="queued")
            + repository.list(runs, status="provisioning")
            + repository.list(runs, status="running"),
            "deployment": repository.list(deployments, status="queued")
            + repository.list(deployments, status="provisioning")
            + repository.list(deployments, status="running"),
        }

    @app.get("/api/internal/v1/workloads/{kind}/{identity}")
    def internal_workload(kind: Literal["training", "deployment"], identity: str, request: Request):
        internal(request)
        value = workload(kind, identity)
        value["allocations"] = repository.list(allocations, owner_id=identity)
        if kind == "training":
            value["currentAttempt"] = repository.attempt(identity, value["attempt"])
        return value

    @app.post("/api/internal/v1/workloads/{kind}/{identity}/reserve")
    def reserve(
        kind: Literal["training", "deployment"], identity: str, request: Request, body: Quote
    ):
        internal(request)
        allocation = repository.reserve(
            identity, kind, body.model_dump(by_alias=True, exclude_none=True)
        )
        objects.ledger(allocation)
        return allocation_view(allocation)

    @app.post("/api/internal/v1/workloads/{kind}/{identity}/project")
    def project(
        kind: Literal["training", "deployment"], identity: str, request: Request, body: Projection
    ):
        internal(request)
        repository.project(
            "TrainingRun" if kind == "training" else "ModelDeployment",
            identity,
            body.status,
            body.phase,
            body.generation,
        )
        return {"ok": True}

    @app.post("/api/internal/v1/workloads/training/{identity}/recover")
    def recover(identity: str, request: Request):
        internal(request)
        run = repository.one(runs, identity)
        if run["attempt"] > 1:
            raise Conflict("Automatic recovery already used")
        records = repository.list(artifacts, run_id=identity, kind="training_checkpoint")
        if not records:
            raise Conflict("No remotely committed checkpoint is available")
        repository.project("TrainingRun", identity, "interrupted", "training", run["attempt"])
        return repository.action(identity, "resume", "latest")

    @app.get("/api/internal/v1/allocations/{identity}")
    def get_allocation(identity: str, request: Request):
        internal(request)
        return allocation_view(repository.one(allocations, identity))

    @app.post("/api/internal/v1/allocations/{identity}/observe")
    def observe(identity: str, request: Request, body: dict):
        internal(request)
        result = repository.observe_allocation(identity, body)
        objects.ledger(result)
        return allocation_view(result)

    @app.get("/api/worker/v1/{identity}/bootstrap")
    def bootstrap(identity: str, request: Request):
        allocation = worker(request, identity)
        owner = workload(allocation["kind"], allocation["owner_id"])
        result = {
            "allocation": {k: v for k, v in allocation.items() if k not in ("telemetry", "result")},
            "deadline": allocation["deadline"],
            "leaseSeconds": settings.lease_seconds,
            "kind": allocation["kind"],
        }
        if allocation["kind"] == "training":
            current = repository.attempt(owner["id"], allocation["attempt"])
            if "prepare" not in current["outputs"]:
                cached = repository.prepared(owner)
                if cached:
                    current["outputs"] = {
                        **current["outputs"],
                        "prepare": cached["identity"]["preparedOutput"],
                    }
            references = set()

            def collect(value):
                if isinstance(value, dict):
                    for child in value.values():
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)
                elif isinstance(value, str) and value.startswith("art-"):
                    references.add(value)

            collect(current["outputs"])
            if current["checkpoint"]:
                references.add(current["checkpoint"])
            with repository.transaction() as c:
                repository.locked_one(budgets, "project", c)
                repository.assert_live(c, allocation)
                from qps.hosted.database import tombstones

                for artifact_id in references:
                    if c.execute(select(tombstones).where(tombstones.c.id == artifact_id)).first():
                        raise Fenced("Prepared input is being collected")
                    repository.add_reference(c, owner["id"], artifact_id, "experiment")
            result.update(
                run=owner["resolved"]["run"],
                runId=owner["id"],
                attempt=allocation["attempt"],
                outputs=current["outputs"],
                checkpoint=current["checkpoint"],
                artifacts=[
                    {
                        **objects.download_descriptor(a),
                        "sourceRun": repository.artifact_recipe(a),
                    }
                    for a in sorted(references)
                ],
            )
        else:
            model = repository.one(models, owner["model_version_id"])
            result.update(
                run=model["snapshot"]["servingRun"],
                deployment=owner,
                artifacts=[objects.download_descriptor(owner["spec"]["artifactId"])],
            )
        return result

    @app.post("/api/worker/v1/{identity}/heartbeat")
    def heartbeat(identity: str, request: Request, body: Heartbeat):
        allocation = worker(request, identity, allow_completed=True)
        if len(canonical(body.telemetry)) > 16384:
            raise ValueError("Telemetry payload is too large")
        result = repository.heartbeat(allocation, body.phase, body.elapsed_seconds, body.telemetry)
        objects.ledger(repository.one(allocations, identity))
        return result

    @app.post("/api/worker/v1/{identity}/events")
    def ingest(identity: str, request: Request, body: WorkerBatch):
        allocation = worker(request, identity)
        return {"sequence": repository.ingest(allocation, body)}

    @app.post("/api/worker/v1/{identity}/phases")
    def phase(identity: str, request: Request, body: PhaseOutput):
        allocation = worker(request, identity)
        result = repository.phase_output(allocation, body.phase, body.output)
        if body.phase == "evaluation":
            repository.promote_model(allocation["owner_id"], objects)
        return result

    @app.post("/api/worker/v1/{identity}/uploads")
    def upload(identity: str, request: Request, body: dict):
        return objects.start(worker(request, identity), body)

    @app.post("/api/worker/v1/{identity}/uploads/{upload_id}/commit")
    def commit(identity: str, upload_id: str, request: Request):
        result = objects.commit(worker(request, identity), upload_id)
        return {"artifactId": result["id"], "manifestSha": result["manifest_sha"]}

    @app.post("/api/worker/v1/{identity}/result")
    def finish(identity: str, request: Request, body: Result):
        repository.finish_worker(worker(request, identity), body.model_dump())
        return {"ok": True}

    @app.post("/api/worker/v1/{identity}/ready")
    def worker_ready(identity: str, request: Request, body: dict):
        allocation = worker(request, identity)
        if allocation["kind"] != "deployment":
            raise Conflict("Only serving workers report model readiness")
        owner = repository.one(deployments, allocation["owner_id"])
        artifact = repository.one(artifacts, owner["spec"]["artifactId"])
        if (
            body.get("artifact_id") != artifact["id"]
            or body.get("manifest_sha") != artifact["manifest_sha"]
        ):
            raise Conflict("Serving worker loaded the wrong artifact")
        repository.observe_allocation(
            identity, {"ready": True, "manifest_sha": artifact["manifest_sha"]}
        )
        return {"ok": True}

    @app.get("/metrics", include_in_schema=False)
    def prometheus():
        from fastapi.responses import PlainTextResponse

        values = [
            "# HELP qps_allocations Active owned compute allocations",
            "# TYPE qps_allocations gauge",
            f"qps_allocations {repository.budget()['activeAllocations']}",
        ]
        for allocation in repository.list(allocations):
            if allocation["state"] == "terminated":
                continue
            stale = not allocation["heartbeat"] or time.time() - allocation["heartbeat"] > 45
            dimensions = {
                "run": allocation["owner_id"],
                "kind": allocation["kind"],
                "attempt": str(allocation["attempt"]),
                "allocation": allocation["id"],
                "worker_group": allocation["worker_group"] or allocation["id"],
            }

            def labels(**extra):
                return ",".join(
                    k + "=" + json.dumps(str(v)) for k, v in {**dimensions, **extra}.items()
                )

            base = labels()
            values.extend(
                [
                    f"qps_worker_telemetry_stale{{{base}}} {int(stale)}",
                    f"qps_active_gpu_count{{{base}}} {allocation['gpu_count']}",
                    f"qps_worker_heartbeat_age_seconds{{{base}}} {max(0, time.time() - (allocation['heartbeat'] or allocation['created']))}",
                    f"qps_worker_phase{{{labels(phase=allocation['phase'])}}} 1",
                ]
            )
            values.append(
                f"qps_gpu_telemetry_available{{{base}}} {int(not stale and allocation['telemetry'].get('available', False))}"
            )
            if not stale:
                for gpu in allocation["telemetry"].get("gpus", []):
                    gpu_labels = labels(
                        gpu=gpu.get("uuid", "unavailable"),
                        device=gpu["index"],
                        rank=gpu.get("rank") if gpu.get("rank") is not None else "idle",
                    )
                    for field in (
                        "utilization_percent",
                        "memory_used_bytes",
                        "memory_total_bytes",
                        "temperature_celsius",
                        "power_watts",
                    ):
                        number = gpu.get(field)
                        values.append(
                            f"qps_gpu_{field}{{{gpu_labels}}} {number if isinstance(number, (int, float)) else 'NaN'}"
                        )
            health = {r["rank"]: r for r in allocation["telemetry"].get("rank_health", [])}
            if allocation["phase"] == "training":
                for rank in range(allocation["gpu_count"]):
                    record = health.get(rank)
                    healthy = not stale and record is not None and time.time() - record["time"] < 45
                    values.append(f"qps_rank_healthy{{{labels(rank=rank)}}} {int(healthy)}")
            if allocation["kind"] == "training":
                with repository.connection() as c:
                    event = (
                        c.execute(
                            select(events)
                            .where(
                                (events.c.run_id == allocation["owner_id"])
                                & (events.c.attempt == allocation["attempt"])
                                & (events.c.kind == "train_step")
                            )
                            .order_by(events.c.id.desc())
                            .limit(1)
                        )
                        .mappings()
                        .first()
                    )
                if event:
                    for field in (
                        "loss",
                        "learning_rate",
                        "grad_norm",
                        "global_step",
                        "tokens_seen",
                        "supervised_tokens_seen",
                    ):
                        number = event["payload"].get(field)
                        if isinstance(number, (int, float)):
                            values.append(f"qps_train_{field}{{{base}}} {number}")
        return PlainTextResponse("\n".join(values) + "\n")

    async def proxy(request, route):
        auth.owner(request)
        if request.method == "GET":
            records = repository.list(deployments, status="running")
            return {
                "object": "list",
                "data": [
                    {
                        "id": d["name"],
                        "object": "model",
                        "owned_by": "qps",
                        "artifact_id": d["spec"]["artifactId"],
                    }
                    for d in records
                ],
            }
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(422, "Request body must be an object")
        model = body.get("model")
        matches = [d for d in repository.list(deployments, name=model) if d["desired"] == "run"]
        if not matches:
            raise HTTPException(404, "Unknown model deployment")
        deployment = matches[0]
        choices = [
            a
            for a in repository.list(allocations, owner_id=deployment["id"])
            if a["generation"] == deployment["generation"]
            and a["ready"]
            and a["state"] != "terminated"
        ]
        if not choices:
            raise HTTPException(503, "Deployment is not ready")
        allocation = choices[0]
        if not allocation["endpoint"]:
            raise HTTPException(503, "Worker endpoint is not available")
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(90, connect=10), trust_env=False, follow_redirects=False
        )
        try:
            upstream = await until_disconnect(
                request,
                client.send(
                    client.build_request(
                        "POST",
                        allocation["endpoint"] + route,
                        json=body,
                        headers={"Authorization": "Bearer " + repository.token(allocation["id"])},
                    ),
                    stream=True,
                ),
            )
        except BaseException:
            await client.aclose()
            raise
        if upstream.status_code != 200 or not body.get("stream"):
            try:
                content = await until_disconnect(request, upstream.aread())
                return Response(
                    content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            finally:
                await upstream.aclose()
                await client.aclose()

        async def chunks():
            try:
                async for chunk in upstream.aiter_raw():
                    if await request.is_disconnected():
                        break
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            chunks(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    @app.get("/v1/models")
    async def model_list(request: Request):
        return await proxy(request, "/v1/models")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await proxy(request, "/v1/completions")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await proxy(request, "/v1/chat/completions")

    return app


def main():
    import uvicorn

    uvicorn.run(
        create_app(),
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        proxy_headers=True,
        forwarded_allow_ips="*",
        access_log=False,
    )


if __name__ == "__main__":
    main()

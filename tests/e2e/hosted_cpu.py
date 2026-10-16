"""Real HTTP/S3/CPU worker proof without cloud resources. Go lifecycle has separate tests."""

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import boto3
import httpx
import uvicorn
from fastapi.responses import JSONResponse
from moto.server import ThreadedMotoServer
from qps.hosted.api import create_app
from qps.hosted.database import Repository, allocations, artifacts, runs
from qps.hosted.objects import ObjectStore
from qps.hosted.settings import Settings
from qps.specs import RunSpec, load_spec
from qps.store import Store
from qps.util import atomic_json, process_alive


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait(predicate, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise TimeoutError("Hosted CPU acceptance condition timed out")


def run(directory: Path):
    directory = directory / ("proof-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=True)
    logging.getLogger("werkzeug").disabled = True
    from werkzeug.serving import WSGIRequestHandler

    WSGIRequestHandler.log = lambda *args, **kwargs: None
    s3port = port()
    apiport = port()
    relayport = port()
    emulator = ThreadedMotoServer(ip_address="127.0.0.1", port=s3port, verbose=False)
    emulator.start()
    settings = Settings(
        auth_mode="development",
        dev_token="local-owner-" + uuid.uuid4().hex,
        internal_token=uuid.uuid4().hex,
        worker_signing_key=uuid.uuid4().hex,
        public_url=f"http://127.0.0.1:{apiport}",
        s3_endpoint=f"http://127.0.0.1:{s3port}",
        bucket="qps-cpu-" + uuid.uuid4().hex,
        dispatch_enabled=False,
    )
    repo = Repository(settings, create=True)
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        region_name="eu-west-2",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    s3.create_bucket(
        Bucket=settings.bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )
    s3.put_bucket_versioning(
        Bucket=settings.bucket, VersioningConfiguration={"Status": "Enabled"}
    )
    objects = ObjectStore(settings, repo, s3)
    application = create_app(settings, repo, objects)
    outages = {"events": 0}

    @application.middleware("http")
    async def temporary_event_outage(request, call_next):
        if (
            request.url.path.startswith("/api/worker/v1/")
            and request.url.path.endswith("/events")
            and outages["events"] < 2
        ):
            outages["events"] += 1
            return JSONResponse(
                {"detail": "Temporary acceptance outage"}, status_code=503
            )
        return await call_next(request)

    server = uvicorn.Server(
        uvicorn.Config(
            application,
            host="127.0.0.1",
            port=apiport,
            log_level="warning",
            access_log=False,
        )
    )
    threading.Thread(target=server.run, daemon=True).start()
    wait(lambda: server.started)
    client = httpx.Client(
        base_url=settings.public_url,
        headers={"Authorization": "Bearer " + settings.dev_token},
        timeout=60,
        trust_env=False,
    )
    children = []
    open_logs = []

    def start(allocation, name):
        work = directory / name
        env = {
            **os.environ,
            "QPS_API_URL": settings.public_url,
            "QPS_ALLOCATION_ID": allocation["id"],
            "QPS_WORKER_TOKEN": repo.token(allocation["id"]),
            "QPS_WORK_DIR": str(work),
            "QPS_RELAY_PORT": str(relayport),
            "TOKENIZERS_PARALLELISM": "false",
        }
        log = (directory / (name + ".log")).open("w")
        open_logs.append(log)
        process = subprocess.Popen(
            [sys.executable, "-m", "qps.hosted.agent"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append(process)
        repo.observe_allocation(
            allocation["id"],
            {
                "state": "running",
                "pod_id": "local-" + str(process.pid),
                "endpoint": f"http://127.0.0.1:{relayport}",
            },
        )
        return process, work

    def post(path, body):
        response = client.post(path, json=body)
        response.raise_for_status()
        return response.json()

    def submit(name, steps=4, accumulation=2):
        spec = load_spec(
            Path(__file__).resolve().parents[2] / "examples/local/tiny-cpu.yaml"
        )
        spec.metadata.name = name
        spec.spec.training.max_steps = steps
        spec.spec.training.gradient_accumulation_steps = accumulation
        spec.spec.training.checkpoint_every_steps = 2 if steps == 4 else 5
        return post(
            "/api/v1/runs",
            {
                "run": spec.model_dump(mode="json", by_alias=True),
                "computeProfile": "cpu-fixture",
                "idempotencyKey": uuid.uuid4().hex,
            },
        )

    def finish_training(record, allocation, process):
        wait(lambda: repo.one(allocations, allocation["id"])["result"], 180)
        process.wait(timeout=15)
        result = repo.one(allocations, allocation["id"])["result"]
        if result["status"] != "succeeded":
            raise RuntimeError(f"CPU worker failed: {result}; inspect {directory}")
        repo.observe_allocation(allocation["id"], {"state": "terminated"})
        repo.project(
            "TrainingRun",
            record["id"],
            "succeeded",
            "completed",
            repo.one(runs, record["id"])["attempt"],
        )

    try:
        first = submit("hosted-cpu-" + uuid.uuid4().hex[:8])
        first_allocation = repo.reserve(
            first["id"], "training", {"hourlyRate": 0, "gpuType": "cpu"}
        )
        first_process, _first_work = start(first_allocation, "initial")
        finish_training(first, first_allocation, first_process)
        assert outages["events"] == 2
        assert [
            event["payload"]["global_step"]
            for event in repo.event_page(first["id"], kinds=("train_step",))
        ] == [1, 2, 3, 4]
        first_outputs = repo.attempt(first["id"], 1)["outputs"]
        # A second run consumes the same committed dataset and parent identities.
        second = submit(
            "hosted-recovery-" + uuid.uuid4().hex[:8], steps=80, accumulation=16
        )
        second_allocation = repo.reserve(
            second["id"], "training", {"hourlyRate": 0, "gpuType": "cpu"}
        )
        interrupted, interrupted_work = start(second_allocation, "interrupted")
        wait(
            lambda: repo.list(
                artifacts, run_id=second["id"], kind="training_checkpoint"
            ),
            180,
        )
        interrupted.kill()
        interrupted.wait(timeout=5)
        journal = Store(interrupted_work / "journal")
        wait(
            lambda: (
                not process_alive(
                    json.loads(journal.run(second["id"])["owner"] or "null")
                )
            ),
            15,
        )
        repo.observe_allocation(second_allocation["id"], {"state": "terminated"})
        repo.project("TrainingRun", second["id"], "interrupted", "training", 1)
        resumed = repo.action(second["id"], "resume", "latest")
        checkpoint = repo.attempt(second["id"], 2)["checkpoint"]
        checkpoint_step = repo.one(artifacts, checkpoint)["manifest"]["metadata"][
            "global_step"
        ]
        replacement = repo.reserve(
            second["id"], "training", {"hourlyRate": 0, "gpuType": "cpu"}
        )
        resumed_process, _resumed_work = start(replacement, "replacement")
        finish_training(resumed, replacement, resumed_process)
        resumed_outputs = repo.attempt(second["id"], 2)["outputs"]
        assert (
            first_outputs["prepare"]["dataset"] == resumed_outputs["prepare"]["dataset"]
        )
        assert (
            first_outputs["prepare"]["parent"] == resumed_outputs["prepare"]["parent"]
        )
        records = repo.event_page(second["id"], 0, 10000)
        resume_events = [e for e in records if e["kind"] == "resume_verified"]
        assert (
            resume_events
            and resume_events[-1]["payload"]["global_step"] == checkpoint_step
        )
        steps = [
            e["payload"]["global_step"]
            for e in records
            if e["kind"] == "train_step" and e["attempt"] == 2
        ]
        assert steps[0] == checkpoint_step + 1 and steps[-1] == 80
        artifact = repo.one(artifacts, first_outputs["export"]["model"])
        deployment = post(
            "/api/v1/deployments",
            {
                "name": "hosted-cpu-serving",
                "artifactId": artifact["id"],
                "computeProfile": "cpu-fixture",
                "idempotencyKey": uuid.uuid4().hex,
            },
        )
        serving = repo.reserve(
            deployment["id"], "deployment", {"hourlyRate": 0, "gpuType": "cpu"}
        )
        serving_process, serving_work = start(serving, "serving")
        wait(lambda: repo.one(allocations, serving["id"])["ready"], 120)
        repo.project("ModelDeployment", deployment["id"], "running", "serving", 1)
        body = {
            "model": "hosted-cpu-serving",
            "messages": [{"role": "user", "content": "What is 2 plus 3?"}],
            "max_tokens": 8,
            "temperature": 0,
        }
        answer = post("/v1/chat/completions", body)["choices"][0]["message"]["content"]
        streamed = ""
        done = False
        with client.stream(
            "POST", "/v1/chat/completions", json={**body, "stream": True}
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line == "data: [DONE]":
                    done = True
                elif line.startswith("data: "):
                    streamed += (
                        json.loads(line[6:])["choices"][0]
                        .get("delta", {})
                        .get("content", "")
                    )
        assert done and streamed == answer
        from qps.data import prompt_ids
        from qps.ml import generate, load_model, tokenizer_from

        runtime = RunSpec.model_validate_json(
            (serving_work / "runtime-spec.json").read_text()
        ).spec
        model_dir = serving_work / "artifacts" / artifact["id"] / "files"
        tokenizer = tokenizer_from(model_dir)
        model = load_model(model_dir, runtime)
        offline = generate(
            model, tokenizer, prompt_ids(tokenizer, body["messages"]), 8
        )["text"]
        assert answer == offline
        # Exercise cancellation through the complete gateway/relay/server chain.
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={**body, "max_tokens": 128, "stream": True},
        ) as response:
            response.raise_for_status()
            next(response.iter_bytes())
        wait(
            lambda: client.post("/v1/chat/completions", json=body).status_code == 200,
            10,
        )
        post("/api/v1/deployments/" + deployment["id"] + "/stop", None)
        serving_process.terminate()
        serving_process.wait(timeout=15)
        repo.observe_allocation(serving["id"], {"state": "terminated"})
        repo.project("ModelDeployment", deployment["id"], "cancelled", "stopped", 1)
        post("/api/v1/deployments/" + deployment["id"] + "/restart", None)
        restarted = repo.reserve(
            deployment["id"], "deployment", {"hourlyRate": 0, "gpuType": "cpu"}
        )
        restarted_process, _ = start(restarted, "serving-restarted")
        wait(lambda: repo.one(allocations, restarted["id"])["ready"], 120)
        repo.project("ModelDeployment", deployment["id"], "running", "serving", 2)
        assert (
            post("/v1/chat/completions", body)["choices"][0]["message"]["content"]
            == answer
        )
        restarted_process.terminate()
        restarted_process.wait(timeout=15)
        repo.observe_allocation(restarted["id"], {"state": "terminated"})
        repo.project("ModelDeployment", deployment["id"], "cancelled", "stopped", 2)
        backup = repo.export_metadata()
        atomic_json(directory / "metadata-backup.json", backup)
        restored = Repository(settings, create=True)
        restored.restore_metadata(backup)
        assert restored.one(runs, second["id"])["status"] == "succeeded"
        report = {
            "status": "passed",
            "directory": str(directory),
            "gpu_validated": False,
            "kubernetes_validated": False,
            "storage": "HTTP S3 emulator",
            "metadata": "SQLite test adapter; PostgreSQL acceptance is separate",
            "initialRun": first["id"],
            "recoveredRun": second["id"],
            "checkpointStep": checkpoint_step,
            "resumedFirstStep": steps[0],
            "resumedLastStep": steps[-1],
            "sharedDataset": first_outputs["prepare"]["dataset"],
            "artifactId": artifact["id"],
            "httpOfflineParity": True,
            "streamingParity": True,
            "disconnectReleasedCapacity": True,
            "serverRestart": True,
            "backupRestore": True,
            "temporaryConnectivityLoss": True,
            "paidResources": False,
        }
        atomic_json(directory / "report.json", report)
        return report
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in open_logs:
            log.close()
        client.close()
        server.should_exit = True
        emulator.stop()


if __name__ == "__main__":
    print(
        json.dumps(
            run(Path(sys.argv[1] if len(sys.argv) > 1 else ".qps/hosted-cpu")), indent=2
        )
    )

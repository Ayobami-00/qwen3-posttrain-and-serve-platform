"""Browser fixture with real API/agent/S3; a fake coordinator replaces Kubernetes only."""

import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import boto3
import uvicorn
from moto.server import ThreadedMotoServer
from qps.hosted.api import create_app
from qps.hosted.database import Repository, allocations, deployments, runs
from qps.hosted.objects import ObjectStore
from qps.hosted.settings import Settings
from werkzeug.serving import WSGIRequestHandler


def main():
    directory = Path(".qps/browser")
    directory.mkdir(parents=True, exist_ok=True)
    logging.getLogger("werkzeug").disabled = True
    WSGIRequestHandler.log = lambda *a, **k: None
    emulator = ThreadedMotoServer(ip_address="127.0.0.1", port=15000, verbose=False)
    emulator.start()
    settings = Settings(
        auth_mode="development",
        dev_token="browser-test-owner-token",
        internal_token="browser-test-internal-token-32-characters",
        worker_signing_key="browser-test-worker-signing-key-32-characters",
        s3_endpoint="http://127.0.0.1:15000",
        dispatch_enabled=False,
        upload_url_seconds=10,
        publication_grace_seconds=2,
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
    stop = threading.Event()
    children = {}

    def controller():
        while not stop.wait(0.25):
            for kind, table in (("training", runs), ("deployment", deployments)):
                for owner in repo.list(table):
                    generation = owner.get("attempt", owner.get("generation"))
                    name = "TrainingRun" if kind == "training" else "ModelDeployment"
                    found = [
                        a
                        for a in repo.list(allocations, owner_id=owner["id"])
                        if a["generation"] == generation
                    ]
                    if not found and owner["status"] == "queued":
                        a = repo.reserve(
                            owner["id"], kind, {"hourlyRate": 0, "gpuType": "cpu"}
                        )
                        work = directory / a["id"]
                        log = (directory / (a["id"] + ".log")).open("w")
                        env = os.environ | {
                            "QPS_API_URL": settings.public_url,
                            "QPS_ALLOCATION_ID": a["id"],
                            "QPS_WORKER_TOKEN": repo.token(a["id"]),
                            "QPS_WORK_DIR": str(work),
                            "QPS_RELAY_PORT": "18000",
                            "TOKENIZERS_PARALLELISM": "false",
                        }
                        child = subprocess.Popen(
                            [sys.executable, "-m", "qps.hosted.agent"],
                            env=env,
                            stdout=log,
                            stderr=log,
                            start_new_session=True,
                        )
                        children[a["id"]] = (child, log)
                        repo.observe_allocation(
                            a["id"],
                            {
                                "state": "running",
                                "pod_id": "local-" + str(child.pid),
                                "endpoint": "http://127.0.0.1:18000",
                            },
                        )
                        repo.project(
                            name, owner["id"], "running", "preparing", generation
                        )
                    elif found:
                        a = found[-1]
                        if a["state"] == "terminated":
                            continue
                        child, _ = children[a["id"]]
                        if (
                            owner["desired"] != "run"
                            or a["result"]
                            or child.poll() is not None
                        ):
                            if child.poll() is None:
                                child.terminate()
                                child.wait(timeout=15)
                            repo.observe_allocation(a["id"], {"state": "terminated"})
                            result = (
                                "cancelled"
                                if owner["desired"] != "run"
                                else (a["result"] or {}).get("status", "failed")
                            )
                            repo.project(
                                name, owner["id"], result, "completed", generation
                            )
                        elif a["heartbeat"]:
                            repo.project(
                                name, owner["id"], "running", a["phase"], generation
                            )

    threading.Thread(target=controller, daemon=True).start()
    try:
        uvicorn.run(
            create_app(settings, repo, objects),
            host="127.0.0.1",
            port=8080,
            access_log=False,
            log_level="warning",
        )
    finally:
        stop.set()
        for child, log in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=15)
            log.close()
        emulator.stop()


if __name__ == "__main__":
    main()

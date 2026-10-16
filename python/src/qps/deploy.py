"""Local deployment processes, readiness validation and identity-safe shutdown."""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time

import httpx

from qps.artifacts import verify_artifact
from qps.specs import Metadata
from qps.util import canonical, file_lock, now, process_alive, process_identity, signal_process


def create(store, run_name, name, bind="127.0.0.1", port=8000, startup_timeout=120):
    Metadata(name=name)
    if bind != "127.0.0.1":
        raise ValueError("v0.1 serves on IPv4 loopback; use an SSH tunnel for remote access")
    run = store.run(run_name)
    if run["status"] != "succeeded":
        raise ValueError("Deploy requires a run with completed evaluation")
    artifact_id = store.output(run["id"], "export")["model"]
    verify_artifact(store, artifact_id)
    directory = store.root / "deployments"
    directory.mkdir(exist_ok=True)
    with file_lock(directory / f"{name}.lock", blocking=False):
        try:
            previous = store.deployment(name)
        except ValueError:
            previous = None
        if previous and process_alive(
            json.loads(previous["identity"]) if previous["identity"] else None
        ):
            raise ValueError("Deployment name is already running")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((bind, port))
            sock.listen(16)
            actual_port = sock.getsockname()[1]
            url = f"http://{bind}:{actual_port}"
            log_path = directory / f"{name}.log"
            command = [
                sys.executable,
                "-m",
                "qps.server",
                "--state",
                str(store.root),
                "--artifact",
                artifact_id,
                "--name",
                name,
                "--fd",
                str(sock.fileno()),
            ]
            with log_path.open("a") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(sock.fileno(),),
                    env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
                )
            identity = process_identity(process.pid)
        finally:
            sock.close()
        profile = json.loads(run["spec"])["spec"]["execution"]["profile"]
        try:
            with store.connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO deployments VALUES(?,?,?,?,?,?,?,?)",
                    (
                        name,
                        artifact_id,
                        "starting",
                        canonical(identity),
                        url,
                        profile,
                        str(log_path),
                        now(),
                    ),
                )
        except BaseException:
            signal_process(identity, signal.SIGTERM, group=True)
            process.wait(timeout=5)
            raise
        try:
            deadline = time.monotonic() + startup_timeout
            with httpx.Client(trust_env=False, timeout=0.5) as client:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Serving process exited; see {log_path}\n{log_path.read_text()[-4000:]}"
                        )
                    try:
                        result = client.get(url + "/readyz")
                        if (
                            result.status_code == 200
                            and result.json()["artifact_id"] == artifact_id
                        ):
                            with store.connect() as db:
                                db.execute(
                                    "UPDATE deployments SET state='ready' WHERE name=?", (name,)
                                )
                            return store.deployment(name)
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
            raise TimeoutError("Deployment readiness deadline exceeded")
        except BaseException:
            stop(store, name)
            with store.connect() as db:
                db.execute("UPDATE deployments SET state='failed' WHERE name=?", (name,))
            raise


def stop(store, name):
    deployment = store.deployment(name)
    identity = json.loads(deployment["identity"]) if deployment["identity"] else None
    signal_process(identity, signal.SIGTERM, group=True)
    deadline = time.monotonic() + 8
    while process_alive(identity) and time.monotonic() < deadline:
        time.sleep(0.1)
    signal_process(identity, signal.SIGKILL, group=True)
    if identity:
        with contextlib.suppress(ChildProcessError, ProcessLookupError):
            os.waitpid(identity["pid"], os.WNOHANG)
    with store.connect() as db:
        db.execute("UPDATE deployments SET state='stopped' WHERE name=?", (name,))


def evaluate(store, name):
    from qps.data import final_answer, question_messages

    deployment = store.deployment(name)
    directory, manifest = verify_artifact(store, deployment["artifact_id"])
    run_id = manifest["run_id"]
    reference = None
    for record in reversed(store.artifacts(run_id)):
        if record["kind"] == "evaluation_report":
            report_directory, _ = verify_artifact(store, record["id"])
            report = json.loads((report_directory / "report.json").read_text())
            if report["artifact_id"] == deployment["artifact_id"]:
                reference = report
                break
    if reference is None:
        raise ValueError(
            "No completed offline evaluation exists for this deployment's exact artifact"
        )
    data_directory, _ = verify_artifact(store, reference["dataset_id"])
    predictions = []
    with httpx.Client(trust_env=False, timeout=120) as client:
        ready = client.get(deployment["url"] + "/readyz")
        ready.raise_for_status()
        if ready.json()["artifact_id"] != deployment["artifact_id"]:
            raise ValueError("HTTP server is serving a different artifact")
        rows = [
            json.loads(line)
            for line in (data_directory / "validation.jsonl").read_text().splitlines()
        ]
        for row, expected in zip(rows, reference["predictions"], strict=True):
            response = client.post(
                deployment["url"] + "/v1/chat/completions",
                json={
                    "model": name,
                    "messages": question_messages(row["question"]),
                    "max_tokens": reference["profile"]["max_new_tokens"],
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            value = response.json()
            text = value["choices"][0]["message"]["content"]
            if (
                text != expected["text"]
                or value["choices"][0]["finish_reason"] != expected["finish_reason"]
            ):
                raise ValueError(f"HTTP/offline generation mismatch for {row['problem_id']}")
            if value["usage"]["completion_tokens"] != expected["completion_tokens"]:
                raise ValueError("HTTP/offline token count mismatch")
            parsed = final_answer(text)
            predictions.append(
                {
                    "problem_id": row["problem_id"],
                    "text": text,
                    "correct": parsed is not None and str(parsed) == row["reference_answer"],
                }
            )
    result = {
        "artifact_id": deployment["artifact_id"],
        "http_offline_parity": True,
        "count": len(predictions),
        "correct": sum(row["correct"] for row in predictions),
        "predictions": predictions,
    }
    from qps.util import atomic_json

    atomic_json(store.root / "deployments" / f"{name}-evaluation.json", result)
    return result

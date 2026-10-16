"""Disposable worker: local journal, durable uploads, fenced leases and reference serving."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import httpx

from qps.artifacts import Artifacts
from qps.hosted.objects import download_artifact
from qps.specs import RunSpec
from qps.store import Store
from qps.util import atomic_json, canonical, fingerprint, process_identity, signal_process


def gpu_telemetry():
    try:
        import pynvml as nv

        nv.nvmlInit()
        values = []
        for index in range(nv.nvmlDeviceGetCount()):
            handle = nv.nvmlDeviceGetHandleByIndex(index)

            def measure(call):
                try:
                    return call()
                except nv.NVMLError:
                    return None

            memory = nv.nvmlDeviceGetMemoryInfo(handle)
            values.append(
                {
                    "index": index,
                    "uuid": nv.nvmlDeviceGetUUID(handle),
                    "name": nv.nvmlDeviceGetName(handle),
                    "utilization_percent": measure(
                        lambda: nv.nvmlDeviceGetUtilizationRates(handle).gpu
                    ),
                    "memory_used_bytes": memory.used,
                    "memory_total_bytes": memory.total,
                    "temperature_celsius": measure(
                        lambda: nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
                    ),
                    "power_watts": measure(lambda: nv.nvmlDeviceGetPowerUsage(handle) / 1000),
                }
            )
        nv.nvmlShutdown()
        return {"available": True, "gpus": values}
    except Exception:
        return {"available": False, "gpus": [], "reason": "NVML unavailable"}


def map_gpu_ranks(telemetry, parallelism, phase, health):
    def identity(value):
        return str(value).lower().removeprefix("gpu-")

    ranks = {identity(r["gpu_uuid"]): r for r in health if r.get("gpu_uuid")}
    for gpu in telemetry["gpus"]:
        observed = ranks.get(identity(gpu["uuid"]))
        if parallelism == "fsdp2" and phase == "training" and observed:
            gpu["rank"], gpu["cuda_device"] = observed["rank"], observed["device"]
        else:
            gpu["rank"] = 0 if len(telemetry["gpus"]) == 1 else None
            gpu["cuda_device"] = 0 if len(telemetry["gpus"]) == 1 else None


class Agent:
    def __init__(self, api_url, identity, token, work: Path):
        self.api_url = api_url.rstrip("/")
        self.id = identity
        self.token = token
        self.work = work.resolve()
        self.api = httpx.Client(
            base_url=self.api_url + "/api/worker/v1/" + identity,
            headers={"Authorization": "Bearer " + token},
            timeout=60,
            follow_redirects=False,
            trust_env=False,
        )
        # Never forward the worker bearer token to S3.
        self.transfer = httpx.Client(timeout=120, follow_redirects=False, trust_env=False)
        self.stop = threading.Event()
        self.process = None
        self.started = time.monotonic()
        self.lease_until = time.time() + 90
        self.deadline = time.time() + 2700
        self.store = None
        self.run_id = None
        self.phase = "bootstrap"
        self.cursor = 0
        self.published = set()
        self.phase_hashes = {}
        self.log_offsets = {}
        self.provenance_emitted = False
        self.compatibility = None
        self.gpu_count = 0
        self.parallelism = "single"

    def request(self, method, path, **kwargs):
        delay = 0.25
        while not self.stop.is_set():
            try:
                response = self.api.request(method, path, **kwargs)
                if response.status_code in (401, 403, 409):
                    raise RuntimeError("Worker lease was rejected or superseded")
                if response.status_code < 500:
                    if response.status_code >= 400:
                        raise RuntimeError(f"Worker protocol rejected HTTP {response.status_code}")
                    return response.json()
            except (httpx.TransportError, httpx.TimeoutException):
                pass
            if time.time() >= min(self.lease_until, self.deadline):
                self.stop.set()
                raise RuntimeError("Worker lease expired while contacting the control plane")
            self.stop.wait(delay)
            delay = min(delay * 2, 2)
        raise RuntimeError("Worker was cancelled")

    def heartbeat(self):
        while not self.stop.is_set():
            try:
                if self.store and self.run_id:
                    self.phase = self.store.run(self.run_id)["phase"]
                telemetry = gpu_telemetry()
                health = []
                if self.store and self.run_id:
                    attempt_dir = self.store.directory(self.run_id) / "attempts" / str(self.attempt)
                    for rank in range(self.gpu_count):
                        path = attempt_dir / f"rank-{rank}-health.json"
                        if path.exists():
                            record = json.loads(path.read_text())
                            record["stale"] = time.time() - record["time"] > 45
                            health.append(record)
                map_gpu_ranks(telemetry, self.parallelism, self.phase, health)
                telemetry["rank_health"] = health
                telemetry["expected_world_size"] = (
                    self.gpu_count if self.parallelism == "fsdp2" else 1
                )
                if self.compatibility is not None:
                    telemetry["compatibility"] = self.compatibility
                if (
                    self.store
                    and self.run_id
                    and self.store.run(self.run_id)["attempt"] == self.attempt
                ):
                    self.store.emit(self.run_id, self.attempt, "worker_telemetry", telemetry)
                reply = self.request(
                    "POST",
                    "/heartbeat",
                    json={
                        "phase": self.phase,
                        "elapsedSeconds": time.monotonic() - self.started,
                        "telemetry": telemetry,
                    },
                )
                self.lease_until = reply["leaseUntil"]
                self.deadline = reply["deadline"]
                if reply["cancel"]:
                    self.stop.set()
            except (RuntimeError, ValueError):
                self.stop.set()
            if self.stop.wait(15):
                break
        if self.process and self.process.poll() is None:
            signal_process(process_identity(self.process.pid), signal.SIGTERM, group=True)

    def watchdog(self):
        while not self.stop.wait(0.5):
            if time.time() > min(self.lease_until, self.deadline):
                self.stop.set()
                if self.process and self.process.poll() is None:
                    signal_process(process_identity(self.process.pid), signal.SIGTERM, group=True)
                break

    def publish(self, record):
        if record["id"] in self.published:
            return
        directory = Path(record["path"])
        manifest = json.loads((directory / "manifest.json").read_text())
        upload = self.request("POST", "/uploads", json=manifest)
        if not upload["committed"]:
            for name, target in upload["files"].items():
                path = directory / "files" / name
                while True:
                    try:
                        with path.open("rb") as stream:
                            response = self.transfer.put(
                                target["url"],
                                headers=target["headers"],
                                content=iter(lambda: stream.read(1024 * 1024), b""),
                            )
                        if response.status_code >= 400:
                            raise httpx.TransportError("Object upload failed")
                        break
                    except httpx.HTTPError:
                        if self.stop.wait(1) or time.time() > min(self.lease_until, self.deadline):
                            raise RuntimeError(
                                "Artifact upload failed before lease expiry"
                            ) from None
            self.request("POST", "/uploads/" + upload["uploadId"] + "/commit")
        self.published.add(record["id"])

    def sync(self):
        if not self.store:
            return
        if not self.provenance_emitted and self.store.run(self.run_id)["attempt"] == self.attempt:
            self.store.emit(
                self.run_id,
                self.attempt,
                "hosted_provenance",
                {
                    "image_digest": os.environ.get("QPS_IMAGE_DIGEST"),
                    "allocation_id": self.id,
                    "lease_generation": self.attempt,
                    "gpu_compatibility": self.compatibility,
                },
            )
            self.provenance_emitted = True
        # Snapshot references before the artifact inventory. The pipeline keeps
        # writing while S3 uploads run; reading phases afterwards can expose a
        # newly completed phase whose adapter was absent from that inventory.
        events = self.store.events(self.run_id, self.cursor)
        with self.store.connect() as connection:
            phases = connection.execute(
                "SELECT phase,output FROM phases WHERE run_id=? AND status='succeeded' ORDER BY attempt",
                (self.run_id,),
            ).fetchall()
        for record in self.store.artifacts(self.run_id):
            self.publish(record)
        for event in events:
            if event["attempt"] == self.attempt:
                self.request(
                    "POST",
                    "/events",
                    json={
                        "events": [
                            {
                                "sequence": event["id"],
                                "time": event["time"],
                                "kind": event["kind"],
                                "payload": event["payload"],
                            }
                        ]
                    },
                )
            self.cursor = event["id"]
        for phase in phases:
            if self.phase_hashes.get(phase["phase"]) != phase["output"]:
                self.request(
                    "POST",
                    "/phases",
                    json={"phase": phase["phase"], "output": json.loads(phase["output"])},
                )
                self.phase_hashes[phase["phase"]] = phase["output"]

    def capture_logs(self):
        directory = self.store.directory(self.run_id) / "attempts" / str(self.attempt)
        for path in directory.glob("*.log"):
            offset = self.log_offsets.get(str(path), 0)
            with path.open("r", errors="replace") as stream:
                stream.seek(offset)
                content = stream.read(8000)
                self.log_offsets[str(path)] = stream.tell()
            if content:
                # Phase processes never receive cloud credentials or signed object URLs.
                self.store.emit(
                    self.run_id, self.attempt, "worker_log", {"phase": path.stem, "text": content}
                )

    def prepare_journal(self, bootstrap):
        self.run_id = bootstrap["runId"]
        self.attempt = bootstrap["attempt"]
        spec = RunSpec.model_validate(bootstrap["run"])
        state = self.work / "journal"
        spec.spec.artifacts.root = self.work / "artifacts"
        if (
            spec.spec.execution.profile == "local-cpu-smoke"
            and "prepare" not in bootstrap["outputs"]
        ):
            from qps.fixtures import create_fixture

            fixture = create_fixture(self.work / "fixture", spec.metadata.name)
            spec.spec.model.id = fixture.spec.model.id
            spec.spec.dataset.id = fixture.spec.dataset.id
        journal = Store(state)
        journal.create_run(spec, run_id=self.run_id)
        self.store = journal
        for descriptor in bootstrap["artifacts"]:
            source_run_id = descriptor["manifest"]["run_id"]
            if source_run_id != self.run_id:
                try:
                    self.store.run(source_run_id)
                except ValueError:
                    linked = RunSpec.model_validate(descriptor["sourceRun"])
                    linked.metadata.name = "linked-" + source_run_id[4:]
                    self.store.create_run(linked, run_id=source_run_id)
            directory = self.work / "artifacts" / descriptor["id"]
            download_artifact(descriptor, directory, self.transfer)
            Artifacts(self.store, self.work / "artifacts")._register(
                directory, descriptor["manifest"]
            )
            self.published.add(descriptor["id"])
        for phase, output in bootstrap["outputs"].items():
            if phase == "prepare":
                output = {
                    **output,
                    "spec_sha": fingerprint(json.loads(self.store.run(self.run_id)["spec"])),
                    "reused_preparation": True,
                }
            self.store.phase(self.run_id, 0, phase, "succeeded", output)
        self.store.update(
            self.run_id,
            attempt=self.attempt - 1,
            status="interrupted" if bootstrap["outputs"] or bootstrap["checkpoint"] else "pending",
            resolved=canonical(bootstrap["outputs"].get("prepare"))
            if bootstrap["outputs"].get("prepare")
            else None,
        )

    def training(self, bootstrap):
        self.prepare_journal(bootstrap)
        action = "resume" if bootstrap["outputs"] or bootstrap["checkpoint"] else "run"
        command = [sys.executable, "-m", "qps.pipeline", str(self.store.root), self.run_id, action]
        if bootstrap["checkpoint"]:
            command.append(bootstrap["checkpoint"])
        # Child phases need no hosted credentials. Only this agent transfers objects.
        environment = {
            key: value
            for key, value in os.environ.items()
            if not any(
                secret in key.upper()
                for secret in (
                    "TOKEN",
                    "SECRET",
                    "PASSWORD",
                    "AWS_ACCESS",
                    "AWS_SESSION",
                    "RUNPOD_API",
                    "WANDB_API",
                    "CREDENTIAL",
                )
            )
        }
        environment.update(
            QPS_IMAGE_DIGEST=os.environ.get("QPS_IMAGE_DIGEST", ""),
            TOKENIZERS_PARALLELISM="false",
            QPS_AGENT_OWNER=canonical(process_identity()),
        )
        with (self.work / "coordinator.log").open("w") as log:
            self.process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            try:
                while self.process.poll() is None and not self.stop.wait(0.5):
                    self.capture_logs()
                    self.sync()
                if self.stop.is_set():
                    signal_process(process_identity(self.process.pid), signal.SIGTERM, group=True)
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self.process.wait(timeout=8)
                    if self.process.poll() is None:
                        signal_process(
                            process_identity(self.process.pid), signal.SIGKILL, group=True
                        )
                    return
                self.capture_logs()
                self.sync()
                result = self.store.run(self.run_id)
                self.request(
                    "POST",
                    "/result",
                    json={
                        "status": result["status"],
                        "reason": ""
                        if result["status"] == "succeeded"
                        else "Pipeline failed; inspect phase events and worker logs",
                    },
                )
            finally:
                if self.process.poll() is None:
                    signal_process(process_identity(self.process.pid), signal.SIGKILL, group=True)

    def serving(self, bootstrap):
        from qps.hosted.relay import start_relay

        descriptor = bootstrap["artifacts"][0]
        directory = self.work / "artifacts" / descriptor["id"]
        download_artifact(descriptor, directory, self.transfer)
        spec = RunSpec.model_validate(bootstrap["run"])
        atomic_json(self.work / "runtime-spec.json", spec.model_dump(mode="json", by_alias=True))
        deployment = bootstrap["deployment"]
        self.phase = "loading"
        command = [
            sys.executable,
            "-m",
            "qps.server",
            "--artifact-directory",
            str(directory),
            "--manifest-sha",
            descriptor["manifestSha"],
            "--spec",
            str(self.work / "runtime-spec.json"),
            "--artifact",
            descriptor["id"],
            "--name",
            deployment["name"],
            "--port",
            "8001",
            "--max-context-tokens",
            str(deployment["spec"]["maxContextTokens"]),
        ]
        with (self.work / "server.log").open("a") as log:
            self.process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                while not self.stop.wait(0.5):
                    if self.process.poll() is not None:
                        raise RuntimeError("Reference server exited during startup")
                    try:
                        response = self.transfer.get("http://127.0.0.1:8001/readyz", timeout=2)
                        if response.status_code == 200:
                            readiness = response.json()
                            break
                    except httpx.HTTPError:
                        pass
                else:
                    return
                if (
                    readiness["artifact_id"] != descriptor["id"]
                    or readiness["manifest_sha"] != descriptor["manifestSha"]
                ):
                    raise RuntimeError("Reference server loaded the wrong artifact")
                relay = start_relay(
                    self.token,
                    self.stop,
                    port=int(os.environ.get("QPS_RELAY_PORT", "8000")),
                    output_limit=deployment["spec"]["maxOutputTokens"],
                )
                self.request("POST", "/ready", json=readiness)
                self.phase = "serving"
                while not self.stop.wait(1):
                    if self.process.poll() is not None:
                        raise RuntimeError("Reference server exited")
                relay.should_exit = True
            finally:
                if self.process.poll() is None:
                    signal_process(process_identity(self.process.pid), signal.SIGTERM, group=True)
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self.process.wait(timeout=8)
                    if self.process.poll() is None:
                        signal_process(
                            process_identity(self.process.pid), signal.SIGKILL, group=True
                        )

    def probe_timeout_seconds(self, workflow):
        # Let the native 120-second process-group timeout emit its diagnostic.
        return 240 if workflow else 150

    def compatibility_probe(self, workflow=False):
        self.phase = "compatibility_probe"
        environment = {
            k: v
            for k, v in os.environ.items()
            if not any(
                word in k.upper()
                for word in (
                    "TOKEN",
                    "SECRET",
                    "PASSWORD",
                    "AWS_ACCESS",
                    "AWS_SESSION",
                    "RUNPOD_API",
                    "WANDB_API",
                    "CREDENTIAL",
                )
            )
        }
        environment["QPS_PROBE_OWNER"] = canonical(process_identity())
        output = self.work / "compatibility.json"
        offset = 0

        def forward_diagnostics():
            nonlocal offset
            with output.open("r", errors="replace") as source:
                source.seek(offset)
                chunk = source.read(16000)
                offset = source.tell()
            if chunk:
                # These offline children have no hosted credentials or object URLs.
                print(chunk, end="", flush=True)

        with output.open("w") as log:
            command = [sys.executable, "-m", "qps.hosted.probe"]
            if workflow:
                command = [
                    sys.executable,
                    "-m",
                    "qps.hosted.workflow_probe",
                    str(self.work / "fsdp-workflow-probe"),
                ]
            elif self.gpu_count == 2:
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=2",
                    "--max-restarts=0",
                    "--module",
                    "qps.hosted.distributed_probe",
                ]
            self.process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            deadline = time.monotonic() + self.probe_timeout_seconds(workflow)
            while self.process.poll() is None and not self.stop.wait(0.25):
                forward_diagnostics()
                if time.monotonic() > deadline:
                    break
            if self.process.poll() is None:
                signal_process(process_identity(self.process.pid), signal.SIGTERM, group=True)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.process.wait(timeout=5)
                if self.process.poll() is None:
                    signal_process(process_identity(self.process.pid), signal.SIGKILL, group=True)
                    self.process.wait(timeout=5)
                forward_diagnostics()
                raise RuntimeError("GPU compatibility probe exceeded its deadline")
            forward_diagnostics()
            if self.process.returncode:
                raise RuntimeError("GPU compatibility probe failed")
        reports = []
        for line in output.read_text().splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(line)
                if isinstance(value, dict) and value.get("passed") is True:
                    reports.append(value)
        if not reports:
            raise ValueError("GPU probe did not produce a successful structured result")
        self.compatibility = reports[-1]
        self.process = None

    def report_bootstrap_failure(self, bootstrap):
        if self.store is not None or bootstrap["kind"] != "training":
            return
        from datetime import UTC, datetime

        path = self.work / "compatibility.json"
        if not path.exists():
            return
        with path.open("r", errors="replace") as stream:
            stream.seek(max(0, path.stat().st_size - 8000))
            diagnostic = stream.read()
        # The probe is offline and runs with the credential-stripped child environment.
        self.request(
            "POST",
            "/events",
            json={
                "events": [
                    {
                        "sequence": 1,
                        "time": datetime.now(UTC).isoformat(),
                        "kind": "worker_error",
                        "payload": {"phase": "compatibility_probe", "log": diagnostic},
                    }
                ]
            },
        )

    def run(self):
        self.work.mkdir(parents=True, exist_ok=True)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self.stop.set())
        bootstrap = self.request("GET", "/bootstrap")
        self.gpu_count = bootstrap["allocation"]["gpu_count"]
        self.parallelism = "fsdp2" if self.gpu_count == 2 else "single"
        self.deadline = bootstrap["deadline"]
        self.lease_until = time.time() + bootstrap["leaseSeconds"]
        threads = [
            threading.Thread(target=self.heartbeat, daemon=True),
            threading.Thread(target=self.watchdog, daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            if bootstrap["allocation"]["profile"] != "cpu-fixture":
                self.compatibility_probe()
                if self.gpu_count == 2:
                    core = self.compatibility
                    self.compatibility_probe(workflow=True)
                    self.compatibility = {**core, "workflow": self.compatibility}
            if bootstrap["kind"] == "training":
                self.training(bootstrap)
            else:
                self.serving(bootstrap)
        except Exception as exc:
            traceback.print_tb(exc.__traceback__)
            print(f"Worker agent failure type: {type(exc).__name__}", file=sys.stderr)
            if not self.stop.is_set():
                with contextlib.suppress(Exception):
                    self.report_bootstrap_failure(bootstrap)
                with contextlib.suppress(Exception):
                    self.request(
                        "POST",
                        "/result",
                        json={
                            "status": "failed",
                            "reason": "Worker agent failed; inspect worker logs",
                        },
                    )
            raise
        finally:
            self.stop.set()
            for thread in threads:
                thread.join(timeout=2)
            self.api.close()
            self.transfer.close()


def main():
    agent = Agent(
        os.environ["QPS_API_URL"],
        os.environ["QPS_ALLOCATION_ID"],
        os.environ["QPS_WORKER_TOKEN"],
        Path(os.environ.get("QPS_WORK_DIR", "/workspace/qps")),
    )
    try:
        agent.run()
    except Exception:
        # Do not log request objects, signed URLs or environment values.
        print("Worker terminated with an error; consult platform phase events.", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

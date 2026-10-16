"""Bounded live acceptance. Downloads small JSON reports only, never model weights."""

import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import httpx
from qps.data import question_messages
from qps.hosted.install import protected_json, read_key
from qps.specs import load_spec


class Acceptance:
    def __init__(self, directory):
        self.directory = directory
        self.installation = json.loads((directory / "installation.json").read_text())
        self.owner = json.loads((directory / "owner-credentials.json").read_text())
        self.state_file = directory / "live-acceptance.json"
        self.state = (
            json.loads(self.state_file.read_text())
            if self.state_file.exists()
            else {
                "runs": [],
                "deployments": [],
                "requests": {},
                "interruptionRequested": None,
            }
        )
        self.client = httpx.Client(
            base_url=self.installation["stack"]["url"], timeout=90, trust_env=False
        )
        self.provider = httpx.Client(
            base_url="https://api.runpod.io/v2",
            timeout=30,
            trust_env=False,
            headers={
                "Authorization": "Bearer "
                + read_key(Path("~/.config/qps/runpod-api-key"))
            },
        )
        self.authenticated = 0

    def save(self):
        protected_json(self.state_file, self.state)

    def request(self, method, path, **kwargs):
        if time.time() - self.authenticated > 3000:
            response = httpx.post(
                "https://cognito-idp.eu-west-2.amazonaws.com/",
                headers={
                    "Content-Type": "application/x-amz-json-1.1",
                    "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
                },
                json={
                    "AuthFlow": "USER_PASSWORD_AUTH",
                    "ClientId": self.installation["stack"]["cognito_client_id"],
                    "AuthParameters": {
                        "USERNAME": self.owner["username"],
                        "PASSWORD": self.owner["password"],
                    },
                },
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError("Owner authentication failed")
            self.client.headers["Authorization"] = (
                "Bearer " + response.json()["AuthenticationResult"]["AccessToken"]
            )
            self.authenticated = time.time()
        response = self.client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Platform {method} {path} returned HTTP {response.status_code}"
            )
        return response.json()

    def small_file(self, identity, filename):
        descriptor = self.request("GET", f"/api/v1/artifacts/{identity}/download")
        info = descriptor["manifest"]["files"][filename]
        if (
            filename not in {"report.json", "validation.jsonl"}
            or info["size"] > 32 * 1024**2
        ):
            raise RuntimeError(
                "Live validation may download only bounded JSON evidence"
            )
        response = httpx.get(descriptor["urls"][filename], timeout=60)
        if (
            response.status_code != 200
            or hashlib.sha256(response.content).hexdigest() != info["sha256"]
        ):
            raise RuntimeError("Evidence checksum verification failed")
        return response.text

    def submit(self, rank, learning_rate):
        key = str(rank)
        if key not in self.state["requests"]:
            spec = load_spec(Path("examples/local/gsm8k-lora.yaml"))
            spec.metadata.name = f"v03-r{rank}-" + uuid.uuid4().hex[:8]
            spec.spec.training.lora.rank = rank
            spec.spec.training.learning_rate = learning_rate
            spec.spec.execution.max_wall_time_seconds = 2700
            self.state["requests"][key] = {
                "run": spec.model_dump(mode="json", by_alias=True),
                "computeProfile": "runpod-qwen3-small",
                "idempotencyKey": uuid.uuid4().hex,
            }
            self.save()
        result = self.request("POST", "/api/v1/runs", json=self.state["requests"][key])
        if result["id"] not in self.state["runs"]:
            self.state["runs"].append(result["id"])
            self.save()
        return result["id"]

    def kill_owned_worker(self, run):
        checkpoints = [
            a for a in run.get("artifacts", []) if a["kind"] == "training_checkpoint"
        ]
        selected = next(
            (
                a
                for a in checkpoints
                if self.request("GET", f"/api/v1/artifacts/{a['id']}")["manifest"][
                    "metadata"
                ]["global_step"]
                == 25
            ),
            None,
        )
        if not selected or run["attempt"] != 1:
            return False
        active = next(
            (
                a
                for a in run["allocations"]
                if a["generation"] == 1 and a["pod_id"] and a["state"] != "terminated"
            ),
            None,
        )
        if not active:
            return False
        response = self.provider.get("/pods/" + active["pod_id"])
        if (
            response.status_code != 200
            or response.json()["name"] != "qps-test-" + active["id"]
        ):
            raise RuntimeError(
                "Fault injection refused: worker ownership was not verified"
            )
        self.state["interruptionRequested"] = {
            "allocationId": active["id"],
            "podId": active["pod_id"],
            "checkpointId": selected["id"],
            "step": 25,
        }
        self.save()
        response = self.provider.delete("/pods/" + active["pod_id"])
        if response.status_code not in (200, 204, 404):
            raise RuntimeError("Controlled worker termination failed")
        return True

    def wait_runs(self):
        deadline = time.time() + 3000
        last = None
        recovery_wait = None
        while time.time() < deadline:
            records = [
                self.request("GET", "/api/v1/runs/" + identity)
                for identity in self.state["runs"]
            ]
            summary = [
                (r["name"], r["status"], r["phase"], r["attempt"]) for r in records
            ]
            if summary != last:
                print("Live experiments:", summary, flush=True)
                last = summary
            if not self.state["interruptionRequested"]:
                self.kill_owned_worker(records[0])
            if any(r["status"] in {"failed", "cancelled"} for r in records):
                raise RuntimeError("A live experiment ended before acceptance")
            if any(
                r["status"] == "interrupted"
                and (
                    index != 0
                    or r["attempt"] != 1
                    or not self.state["interruptionRequested"]
                )
                for index, r in enumerate(records)
            ):
                raise RuntimeError(
                    "An unplanned worker interruption prevented acceptance"
                )
            if records[0]["status"] == "interrupted":
                recovery_wait = recovery_wait or time.time()
                if time.time() - recovery_wait > 120:
                    raise RuntimeError("Automatic checkpoint replacement did not start")
            if all(r["status"] == "succeeded" for r in records) and all(
                a["state"] == "terminated" for r in records for a in r["allocations"]
            ):
                assert (
                    records[0]["attempt"] == 2 and self.state["interruptionRequested"]
                )
                return records
            time.sleep(10)
        raise TimeoutError("Live experiments exceeded their bounded acceptance wait")

    def wait_deployment(self, identity, desired):
        deadline = time.time() + 600
        while time.time() < deadline:
            record = next(
                d
                for d in self.request("GET", "/api/v1/deployments")
                if d["id"] == identity
            )
            if record["status"] == desired:
                return record
            if desired == "running" and record["status"] in {"failed", "interrupted"}:
                raise RuntimeError("Serving allocation failed")
            time.sleep(3)
        raise TimeoutError("Serving lifecycle did not reach the expected state")

    def resource_evidence(self, run):
        cursor = 0
        telemetry = []
        compatibility = []
        resumptions = []
        while True:
            page = self.request(
                "GET", f"/api/v1/runs/{run['id']}/events", params={"after": cursor}
            )
            if not page:
                break
            cursor = page[-1]["id"]
            for event in page:
                if event["kind"] == "worker_telemetry":
                    telemetry.extend(event["payload"].get("gpus", []))
                elif event["kind"] == "hosted_provenance":
                    compatibility.append(event["payload"]["gpu_compatibility"])
                elif event["kind"] == "resume_verified":
                    resumptions.append(event["payload"])
        metrics = self.request("GET", f"/api/v1/runs/{run['id']}/metrics")
        peaks = [m["payload"].get("cuda_peak_allocated_bytes") for m in metrics]
        assert len(compatibility) == run["attempt"] and all(
            p["passed"] for p in compatibility
        )
        return {
            "compatibilityProbes": compatibility,
            "sampleCount": len(telemetry),
            "optimizerResumptions": resumptions,
            "peakSampledDeviceBytes": max(
                (g["memory_used_bytes"] for g in telemetry), default=None
            ),
            "peakTrainingCudaBytes": max(
                (v for v in peaks if v is not None), default=None
            ),
            "allocationSeconds": sum(
                a["ended"] - a["created"] for a in run["allocations"]
            ),
        }

    def verify_forge(self, records):
        import wandb

        expected = [(r["id"], a["number"]) for r in records for a in r["attempts"]]
        deadline = time.time() + 120
        while time.time() < deadline:
            api = wandb.Api(
                api_key=read_key(Path("~/.config/qps/wandb-api-key")),
                overrides={"base_url": "https://api.forge.coreweave.com"},
                timeout=20,
            )
            summaries = []
            try:
                for identity, attempt in expected:
                    run = api.run(f"enfinity7b/qps-platform/{identity}-a{attempt}")
                    summary = dict(run.summary)
                    if summary.get("qps/run_id") != identity or not summary.get(
                        "qps/status"
                    ):
                        break
                    assert run.group == identity
                    if summary["qps/status"] == "succeeded":
                        assert summary.get("train/global_step") == 100
                    summaries.append(
                        {
                            "id": run.id,
                            "group": run.group,
                            "status": summary["qps/status"],
                            "lastGlobalStep": summary.get("train/global_step"),
                            "modelRevision": summary.get("qps/model_revision"),
                            "datasetRevision": summary.get("qps/dataset_revision"),
                        }
                    )
                if len(summaries) == len(expected):
                    assert summaries[-1]["lastGlobalStep"] == 100
                    return summaries
            except wandb.errors.CommError:
                pass
            time.sleep(5)
        raise TimeoutError("Forge did not synchronize all platform attempts")

    def verify(self, records):
        first, second = [
            min(r["allocations"], key=lambda a: a["generation"]) for r in records
        ]
        assert max(first["created"], second["created"]) < min(
            first["ended"], second["ended"]
        ), "The two experiments did not overlap"
        outputs = [
            max(r["attempts"], key=lambda a: a["number"])["outputs"] for r in records
        ]
        assert outputs[0]["prepare"]["dataset"] == outputs[1]["prepare"]["dataset"]
        assert outputs[0]["prepare"]["parent"] == outputs[1]["prepare"]["parent"]
        comparisons = [
            json.loads(self.small_file(o["evaluation"]["comparison"], "report.json"))
            for o in outputs
        ]
        assert all(
            c["baseline"]["count"] == c["candidate"]["count"] == 128
            for c in comparisons
        )
        recovered = self.request("GET", f"/api/v1/runs/{records[0]['id']}/metrics")
        steps = [
            m["payload"]["global_step"]
            for m in recovered
            if m["kind"] == "train_step" and m["attempt"] == 2
        ]
        assert steps == list(range(26, 101))
        second_metrics = self.request("GET", f"/api/v1/runs/{records[1]['id']}/metrics")
        assert [
            m["payload"]["global_step"]
            for m in second_metrics
            if m["kind"] == "train_step"
        ] == list(range(1, 101))
        resources = [self.resource_evidence(r) for r in records]
        restored = resources[0]["optimizerResumptions"]
        assert (
            len(restored) == 1
            and restored[0]["global_step"] == restored[0]["optimizer_min_step"] == 25
        )
        tracking = self.verify_forge(records)
        artifact = outputs[0]["export"]["model"]
        if not self.state["deployments"]:
            result = self.request(
                "POST",
                "/api/v1/deployments",
                json={
                    "name": "v03-serve-" + uuid.uuid4().hex[:8],
                    "artifactId": artifact,
                    "computeProfile": "runpod-qwen3-small",
                    "maxRuntimeSeconds": 1800,
                    "idempotencyKey": uuid.uuid4().hex,
                },
            )
            self.state["deployments"].append(result["id"])
            self.save()
        deployment = self.wait_deployment(self.state["deployments"][0], "running")
        data = [
            json.loads(line)
            for line in self.small_file(
                outputs[0]["prepare"]["dataset"], "validation.jsonl"
            ).splitlines()
        ]
        row = data[0]
        assert len(data) == 128
        expected = next(
            p["text"]
            for p in comparisons[0]["candidate"]["predictions"]
            if p["problem_id"] == row["problem_id"]
        )
        body = {
            "model": deployment["name"],
            "messages": question_messages(row["question"]),
            "max_tokens": 512,
            "temperature": 0,
        }
        assert any(
            m["artifact_id"] == artifact
            for m in self.request("GET", "/v1/models")["data"]
        )
        answer = self.request("POST", "/v1/chat/completions", json=body)["choices"][0][
            "message"
        ]["content"]
        assert answer == expected
        self.state["servingChecks"] = {
            "deploymentId": deployment["id"],
            "artifactId": artifact,
            "httpOfflineParity": True,
        }
        self.save()
        chunks = []
        done = False
        with self.client.stream(
            "POST", "/v1/chat/completions", json=body | {"stream": True}
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line == "data: [DONE]":
                    done = True
                elif line.startswith("data: "):
                    chunks.append(
                        json.loads(line[6:])["choices"][0]
                        .get("delta", {})
                        .get("content", "")
                    )
        assert done and "".join(chunks) == answer
        self.state["servingChecks"]["streamingParity"] = True
        self.save()
        with self.client.stream(
            "POST", "/v1/chat/completions", json=body | {"stream": True}
        ) as response:
            assert response.status_code == 200
            next(response.iter_bytes())
        time.sleep(1)
        self.request("POST", "/v1/chat/completions", json=body | {"max_tokens": 4})
        self.state["servingChecks"]["disconnectReleasedCapacity"] = True
        self.save()
        self.request("POST", f"/api/v1/deployments/{deployment['id']}/stop")
        self.wait_deployment(deployment["id"], "cancelled")
        self.request("POST", f"/api/v1/deployments/{deployment['id']}/restart")
        self.state["servingChecks"]["restartRequested"] = True
        self.save()
        self.wait_deployment(deployment["id"], "running")
        assert (
            self.request("POST", "/v1/chat/completions", json=body)["choices"][0][
                "message"
            ]["content"]
            == answer
        )
        self.state["servingChecks"]["servingRestart"] = True
        self.save()
        return {
            "workflowCorrect": True,
            "concurrentExperiments": True,
            "sharedDataset": outputs[0]["prepare"]["dataset"],
            "artifactId": artifact,
            "optimizerRecovery": {
                "checkpointStep": 25,
                "firstResumedStep": 26,
                "lastStep": 100,
            },
            "httpOfflineParity": True,
            "streamingParity": True,
            "disconnectReleasedCapacity": True,
            "servingRestart": True,
            "resources": resources,
            "forgeAttempts": tracking,
            "quality": [
                {
                    "runId": r["id"],
                    "parent": {
                        k: c["baseline"][k]
                        for k in (
                            "accuracy",
                            "correct",
                            "count",
                            "parse_failures",
                            "truncations",
                        )
                    },
                    "candidate": {
                        k: c["candidate"][k]
                        for k in (
                            "accuracy",
                            "correct",
                            "count",
                            "parse_failures",
                            "truncations",
                        )
                    },
                    "parity": c["parity"],
                    "activeSeconds": c["total_active_seconds"],
                }
                for r, c in zip(records, comparisons, strict=True)
            ],
        }

    def cleanup(self):
        for identity in self.state["deployments"]:
            self.request("POST", f"/api/v1/deployments/{identity}/stop")
        for identity in self.state["runs"]:
            self.request(
                "POST", f"/api/v1/runs/{identity}/actions", json={"action": "cancel"}
            )
        deadline = time.time() + 180
        while time.time() < deadline:
            values = [
                self.request("GET", "/api/v1/runs/" + identity)
                for identity in self.state["runs"]
            ]
            active = [
                a
                for r in values
                for a in r["allocations"]
                if a["state"] != "terminated"
            ]
            deps = self.request("GET", "/api/v1/deployments")
            if not active and all(
                d["status"] == "cancelled"
                for d in deps
                if d["id"] in self.state["deployments"]
            ):
                break
            time.sleep(3)
        else:
            raise RuntimeError("Live GPU cleanup remains unconfirmed")
        allocation_ids = {a["id"] for r in values for a in r["allocations"]}
        pods = self.provider.get("/pods").json()["pods"]
        assert not any(
            p["name"] == "qps-test-" + identity
            for p in pods
            for identity in allocation_ids
        )
        return self.request("GET", "/api/v1/admin/budget")


def main():
    acceptance = Acceptance(Path(".qps/platform/test"))
    result = {"workflowCorrect": False}
    failure = None
    try:
        acceptance.submit(16, 0.0001)
        acceptance.submit(8, 0.0002)
        result = acceptance.verify(acceptance.wait_runs())
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - Cleanup also follows interruption.
        result["failureType"] = type(exc).__name__
        failure = exc
    try:
        result["cleanup"] = acceptance.cleanup()
        result["cleanupConfirmed"] = True
    except Exception as exc:  # noqa: BLE001 - Retain evidence even during a provider outage.
        result["cleanupConfirmed"] = False
        result["cleanupFailureType"] = type(exc).__name__
        failure = failure or exc
    protected_json(
        acceptance.directory / "live-report.json",
        result
        | {
            "runs": acceptance.state["runs"],
            "deployments": acceptance.state["deployments"],
            "retiredDeployments": acceptance.state.get("retiredDeployments", []),
            "servingChecks": acceptance.state.get("servingChecks", {}),
        },
    )
    if failure:
        raise RuntimeError(
            "Live acceptance incomplete; protected evidence retained"
        ) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - Never print credential-bearing HTTP exception strings.
        print("Live acceptance stopped: " + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1) from None

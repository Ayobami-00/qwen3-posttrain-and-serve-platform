"""Explicitly opted-in v0.4 live acceptance. Never downloads model weights."""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from hosted_live import Acceptance
from qps.hosted.install import protected_json
from qps.specs import load_spec


class DistributedAcceptance(Acceptance):
    def __init__(self, directory):
        super().__init__(directory)
        self.state_file = directory / "live-v04.json"
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

    def opt_in(self):
        if os.environ.get("QPS_LIVE_FSDP2") != "1":
            raise RuntimeError("Paid acceptance requires QPS_LIVE_FSDP2=1")
        self.installation = json.loads(
            (self.directory / "installation.json").read_text()
        )
        if self.installation["budgetDeadline"] - time.time() < 2700:
            raise RuntimeError(
                "The unchanged AWS window cannot cover GPU acceptance and cleanup"
            )
        if self.installation["forecast"]["estimate"] >= 20:
            raise RuntimeError("The cumulative AWS forecast must remain below $20")

    def wait_operation(self, operation):
        deadline = min(time.time() + 1000, self.installation["budgetDeadline"])
        last = None
        while time.time() < deadline:
            value = self.request("GET", f"/api/v1/operations/{operation['id']}")
            phase = (value["state"], value["phase"])
            if phase != last:
                print("Cleanup operation", operation["id"], *phase, flush=True)
                last = phase
            if value["state"] == "completed":
                return value
            time.sleep(2)
        raise TimeoutError("Deletion remains pending; inspect the durable operation")

    def inventory(self):
        return {
            name: self.request("GET", "/api/v1/" + name)
            for name in ("runs", "models", "deployments")
        }

    def reset(self):
        before = self.request("GET", "/api/v1/admin/budget")
        inventory = self.inventory()
        protected_json(
            self.directory / "v04-reset-inventory.json",
            {
                name: [
                    {
                        k: v
                        for k, v in r.items()
                        if k in ("id", "name", "status", "artifact_id")
                    }
                    for r in records
                ]
                for name, records in inventory.items()
            },
        )
        for resource in ("runs", "models", "deployments"):
            for item in self.request("GET", "/api/v1/" + resource):
                operation = self.request(
                    "DELETE",
                    f"/api/v1/{resource}/{item['id']}",
                    params={"confirm": item["id"]},
                )
                self.wait_operation(operation)
        after = self.inventory()
        assert all(not records for records in after.values())
        assert self.request("GET", "/api/v1/artifacts") == []
        assert self.request("GET", "/api/v1/datasets")
        budget = self.request("GET", "/api/v1/admin/budget")
        assert budget["estimatedSpentUsd"] >= before["estimatedSpentUsd"]
        protected_json(
            self.directory / "v04-reset-report.json",
            {
                "emptyProductWorkspace": True,
                "spendingBefore": before["estimatedSpentUsd"],
                "spendingAfter": budget["estimatedSpentUsd"],
                "catalogPreserved": True,
            },
        )
        return budget

    def submit(self, rank, learning_rate):
        self.opt_in()
        key = str(rank)
        if key not in self.state["requests"]:
            spec = load_spec(Path("examples/hosted/gsm8k-fsdp2.yaml"))
            spec.metadata.name = f"v04-r{rank}-" + uuid.uuid4().hex[:8]
            spec.spec.training.lora.rank = rank
            spec.spec.training.learning_rate = learning_rate
            self.state["requests"][key] = {
                "run": spec.model_dump(mode="json", by_alias=True),
                "computeProfile": "runpod-qwen3-fsdp2",
                "idempotencyKey": uuid.uuid4().hex,
            }
            self.save()
        result = self.request("POST", "/api/v1/runs", json=self.state["requests"][key])
        if result["id"] not in self.state["runs"]:
            self.state["runs"].append(result["id"])
            self.save()
        return result["id"]

    def kill_owned_worker(self, run):
        active = [
            a
            for a in run["allocations"]
            if a["state"] != "terminated" and a["generation"] == 1
        ]
        if not active:
            return False
        allocation = active[-1]
        if (
            allocation["gpu_count"] != 2
            or allocation["profile"] != "runpod-qwen3-fsdp2"
        ):
            raise RuntimeError(
                "Fault injection requires a verified two-GPU worker group"
            )
        return super().kill_owned_worker(run)

    def status(self):
        records = [
            self.request("GET", "/api/v1/runs/" + identity)
            for identity in self.state["runs"]
        ]
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "status": r["status"],
                "phase": r["phase"],
                "attempt": r["attempt"],
            }
            for r in records
        ]

    def wait_runs(self):
        deadline = min(time.time() + 3000, self.installation["budgetDeadline"] - 120)
        last = None
        while time.time() < deadline:
            records = [
                self.request("GET", "/api/v1/runs/" + identity)
                for identity in self.state["runs"]
            ]
            summary = [
                (r["name"], r["status"], r["phase"], r["attempt"]) for r in records
            ]
            if summary != last:
                print("Distributed experiments:", summary, flush=True)
                last = summary
            if not self.state["interruptionRequested"]:
                self.kill_owned_worker(records[0])
            if any(r["status"] in ("failed", "cancelled") for r in records):
                raise RuntimeError(
                    "A distributed experiment failed; preserve its diagnostics"
                )
            if all(
                r["status"] == "succeeded"
                and all(a["state"] == "terminated" for a in r["allocations"])
                for r in records
            ):
                assert (
                    records[0]["attempt"] == 2 and self.state["interruptionRequested"]
                )
                return records
            time.sleep(2)
        raise TimeoutError(
            "Distributed acceptance exceeded the remaining environment window"
        )

    def verify(self, records):
        self.opt_in()
        for record in records:
            prepared = [
                e
                for e in self.request("GET", f"/api/v1/runs/{record['id']}/events")
                if e["kind"] == "distributed_prepared"
            ]
            assert prepared and all(
                e["payload"]["world_size"] == 2
                and e["payload"]["local_parameter_elements"]
                < e["payload"]["global_parameter_elements"]
                for e in prepared
            )
        published = [
            m
            for m in self.request("GET", "/api/v1/models")
            if m["source_run_id"] == records[0]["id"]
        ]
        assert len(published) == 1
        if not self.state["deployments"]:
            result = self.request(
                "POST",
                "/api/v1/deployments",
                json={
                    "name": "v04-python-serving-" + uuid.uuid4().hex[:6],
                    "modelVersionId": published[0]["id"],
                    "computeProfile": "runpod-qwen3-small",
                    "maxRuntimeSeconds": 900,
                    "idempotencyKey": uuid.uuid4().hex,
                },
            )
            self.state["deployments"].append(result["id"])
            self.save()
        result = super().verify(records)
        result.update(
            version="0.4",
            fsdp2NcclValidated=True,
            modelVersionId=published[0]["id"],
            modelVersions=[m["id"] for m in self.request("GET", "/api/v1/models")],
        )
        protected_json(self.directory / "live-v04-report.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["inventory", "reset", "submit", "status", "wait", "verify", "cleanup"],
    )
    args = parser.parse_args()
    acceptance = DistributedAcceptance(Path(".qps/platform/test"))
    if args.action == "inventory":
        result = {
            k: [{f: r[f] for f in ("id", "name", "status")} for r in v]
            for k, v in acceptance.inventory().items()
        }
    elif args.action == "submit":
        result = [acceptance.submit(16, 0.0001), acceptance.submit(8, 0.0002)]
    elif args.action == "wait":
        result = {"completed": [r["id"] for r in acceptance.wait_runs()]}
    elif args.action == "verify":
        result = acceptance.verify(
            [
                acceptance.request("GET", "/api/v1/runs/" + i)
                for i in acceptance.state["runs"]
            ]
        )
    else:
        result = getattr(acceptance, args.action)()
    print(json.dumps(result, indent=2), flush=True)

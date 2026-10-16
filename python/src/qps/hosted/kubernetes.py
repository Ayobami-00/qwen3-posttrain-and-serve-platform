"""Small Kubernetes transport; PostgreSQL outbox makes delivery restartable."""

import os
from pathlib import Path

import httpx


class Kubernetes:
    def __init__(self, namespace):
        root = Path("/var/run/secrets/kubernetes.io/serviceaccount")
        self.namespace = namespace
        self.client = httpx.Client(
            base_url="https://"
            + os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
            + ":"
            + os.environ.get("KUBERNETES_SERVICE_PORT", "443"),
            verify=str(root / "ca.crt"),
            headers={"Authorization": "Bearer " + (root / "token").read_text().strip()},
            timeout=15,
            trust_env=False,
        )

    def apply(self, kind, identity, generation, action):
        plural = {"TrainingRun": "trainingruns", "ModelDeployment": "modeldeployments"}[kind]
        value = {
            "apiVersion": "qps.dev/v1alpha1",
            "kind": kind,
            "metadata": {
                "name": identity,
                "namespace": self.namespace,
                "labels": {"app.kubernetes.io/part-of": "qps"},
            },
            "spec": {"ownerId": identity, "generation": generation, "action": action},
        }
        response = self.client.patch(
            f"/apis/qps.dev/v1alpha1/namespaces/{self.namespace}/{plural}/{identity}",
            params={"fieldManager": "qps-dispatch", "force": "true"},
            json=value,
            headers={"Content-Type": "application/apply-patch+yaml"},
        )
        response.raise_for_status()

    def remove(self, plural, identity):
        path = f"/apis/qps.dev/v1alpha1/namespaces/{self.namespace}/{plural}/{identity}"
        response = self.client.get(path)
        if response.status_code == 404:
            return
        response.raise_for_status()
        meta = response.json()["metadata"]
        owned_allocation = (
            plural == "computeallocations"
            and response.json().get("spec", {}).get("allocationId") == identity
            and any(
                o.get("kind") in ("TrainingRun", "ModelDeployment")
                for o in meta.get("ownerReferences", [])
            )
        )
        if (
            meta.get("labels", {}).get("app.kubernetes.io/part-of") != "qps"
            and not owned_allocation
        ):
            raise RuntimeError("Kubernetes cleanup requires verified project ownership")
        response = self.client.request(
            "DELETE",
            path,
            json={
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "preconditions": {"uid": meta["uid"]},
                "propagationPolicy": "Foreground",
            },
        )
        if response.status_code != 404:
            response.raise_for_status()
            raise RuntimeError("Waiting for Kubernetes finalizers")


def dispatch(repository, stop):
    transport = Kubernetes(repository.settings.namespace)
    while not stop.wait(0.5):
        record = repository.claim_outbox()
        if not record:
            continue
        try:
            transport.apply(
                record["resource"], record["resource_id"], record["generation"], record["action"]
            )
            repository.ack_outbox(record["id"])
        except (httpx.HTTPError, OSError):
            # The expiring claim is retried; never print headers, tokens or response bodies.
            __import__("logging").getLogger(__name__).warning(
                "Kubernetes dispatch failed; retrying outbox %s", record["id"]
            )

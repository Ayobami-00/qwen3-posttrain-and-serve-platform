import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_hosted_storage import registry as hosted_registry

from qps.hosted.api import create_app

registry = hosted_registry


@pytest.fixture
def client(registry):
    repo, _, _ = registry
    # Object operations are not needed by these authentication/contract checks.
    return TestClient(create_app(repo.settings, repo, objects=object()))


def test_anonymous_summary_does_not_expose_runtime_or_provider_state(client, registry):
    _, run, _ = registry
    response = client.get("/api/v1/runs/" + run["id"])
    assert response.status_code == 200
    value = response.json()
    assert value["model"] == "tiny-qwen3-fixture"
    assert not {"submission", "resolved", "allocations", "attempts", "artifacts"} & value.keys()
    assert client.get("/api/v1/runs/" + run["id"] + "/events").status_code == 401
    assert client.get("/api/v1/admin/budget").status_code == 401
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/chat/completions", json={}).status_code == 401


def test_owner_actions_require_csrf_when_using_cookie(client, registry):
    _, run, _ = registry
    client.cookies.set("qps_dev", "owner")
    route = "/api/v1/runs/" + run["id"] + "/actions"
    assert client.post(route, json={"action": "cancel"}).status_code == 403
    assert (
        client.post(route, json={"action": "cancel"}, headers={"x-csrf-token": "owner"}).status_code
        == 200
    )
    assert client.get("/api/v1/auth/me").json()["csrfToken"] == "owner"


def test_bearer_authorization_and_operator_boundary(client, registry):
    _, run, _ = registry
    headers = {"Authorization": "Bearer owner"}
    assert "submission" in client.get("/api/v1/runs/" + run["id"], headers=headers).json()
    assert client.get("/api/internal/v1/queue", headers=headers).status_code == 401
    assert (
        client.get(
            "/api/internal/v1/queue", headers={"Authorization": "Bearer " + "i" * 32}
        ).status_code
        == 200
    )
    assert client.get("/api/v1/backends").json()[0]["gpuCounts"] == [0, 1]


def test_operator_allocation_shape_matches_shared_fixture(client, registry):
    repo, run, _ = registry
    allocation = repo.reserve(run["id"], "training", {"hourlyRate": 0, "gpuType": "cpu"})
    actual = client.get(
        "/api/internal/v1/allocations/" + allocation["id"],
        headers={"Authorization": "Bearer " + "i" * 32},
    ).json()
    fixture = json.loads(
        (Path(__file__).resolve().parents[2] / "contracts/fixtures/allocation.json").read_text()
    )
    assert actual.keys() == fixture.keys()
    assert actual["gpu_count"] == 0
    assert actual["worker_group"] == allocation["id"]
    assert {key: type(value) for key, value in actual.items()} == {
        key: type(value) for key, value in fixture.items()
    }


def test_public_deployments_exclude_provider_image_and_request_identity(client, registry):
    from sqlalchemy import insert

    from qps.hosted.database import deployments

    repo, _, _ = registry
    with repo.transaction() as connection:
        connection.execute(
            insert(deployments).values(
                id="dep-fixture",
                name="fixture",
                status="running",
                spec={
                    "artifactId": "art-fixture",
                    "backend": "python-transformers",
                    "workerImage": "private-registry/worker@sha256:abc",
                    "idempotencyKey": "private-request",
                },
            )
        )
    assert client.get("/api/v1/deployments").json()[0]["spec"] == {
        "artifactId": "art-fixture",
        "backend": "python-transformers",
    }
    assert (
        "workerImage"
        in client.get("/api/v1/deployments", headers={"Authorization": "Bearer owner"}).json()[0][
            "spec"
        ]
    )


def test_submit_rejects_unsupported_fields_and_duplicate_name(client, registry):
    _, _, submission = registry
    body = submission.model_dump(mode="json", by_alias=True)
    body["idempotencyKey"] = "new-key-123"
    assert client.post("/api/v1/runs", json=body).status_code == 401
    assert (
        client.post(
            "/api/v1/runs", json=body, headers={"Authorization": "Bearer owner"}
        ).status_code
        == 409
    )
    body["run"]["spec"]["training"]["unknownSetting"] = 1
    assert (
        client.post(
            "/api/v1/runs", json=body, headers={"Authorization": "Bearer owner"}
        ).status_code
        == 422
    )

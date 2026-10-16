import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from qps.hosted.install import Installer, PlatformConfig, forecast, protected_json, read_key


def janitor():
    spec = importlib.util.spec_from_file_location(
        "qps_janitor_test", Path(__file__).resolve().parents[2] / "infra/lambda/janitor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_and_lifetime_cannot_expand_approved_scope():
    for key, value in {
        "aws_total_usd": 31,
        "aws_forecast_ceiling_usd": 21,
        "runpod_total_usd": 4,
        "runpod_hourly_usd": 1.01,
        "lifetime_hours": 25,
    }.items():
        with pytest.raises(ValidationError):
            PlatformConfig(account_id="127811062476", **{key: value})
    assert forecast()["estimate"] < 20
    assert forecast()["hardCap"] is False
    assert forecast(12)["estimate"] < forecast(24)["estimate"]
    assert forecast()["hours"] < forecast()["requestedHours"]
    assert forecast()["requestedLifetimeEstimate"] > 20
    assert forecast(gpu_image_bytes=10 * 1024**3)["hours"] < forecast()["hours"]


def test_repeat_apply_keeps_original_deadline_and_image_identity(monkeypatch):
    import time

    installer = object.__new__(Installer)
    deadline = time.time() + 3600
    installer.installation = {
        "stack": {"url": "https://example.invalid", "cluster_project": "qps-test-cluster"},
        "tracking": {"reportUrl": "https://example.invalid/report"},
        "sourceSha": "abc",
        "imagesBuildSha": "abc",
        "budgetDeadline": deadline,
        "images": {"worker": "example@sha256:abc"},
    }
    calls = []
    monkeypatch.setattr(installer, "build", lambda *args: calls.append(args))
    monkeypatch.setattr(installer, "refresh", lambda: None)
    assert installer.apply()["reapplied"]
    assert installer.installation["budgetDeadline"] == deadline
    assert installer.installation["images"]["worker"] == "example@sha256:abc"
    assert calls == [("qps-test-cluster", "cluster-apply")]


def test_installer_changes_can_reuse_images_but_runtime_changes_cannot(monkeypatch):
    installer = object.__new__(Installer)

    def tree(command):
        revision = command[3]
        runtime = "changed" if revision == "runtime-change" else "same"
        return (
            f"100644 blob {runtime}\tpython/src/qps/ml.py\n"
            f"100644 blob {revision}\tpython/src/qps/hosted/install.py\n"
            f"100644 blob {revision}\tpython/src/qps/hosted/operations.py\n"
        )

    monkeypatch.setattr(installer, "command", tree)
    assert installer.image_inputs("old") == installer.image_inputs("installer-change")
    assert installer.image_inputs("old") != installer.image_inputs("runtime-change")


def test_image_recovery_uses_only_successful_recorded_builds(monkeypatch):
    installer = object.__new__(Installer)
    installer.installation = {
        "sourceSha": "new",
        "builds": [{"id": "qps-test-images:owned", "operation": "images"}],
    }
    monkeypatch.setattr(installer, "image_inputs", lambda revision: "same-inputs")
    record = {
        "startTime": "2026-10-16T00:00:00Z",
        "buildStatus": "IN_PROGRESS",
        "environment": {"environmentVariables": [{"name": "QPS_SOURCE_SHA", "value": "old"}]},
    }
    monkeypatch.setattr(installer, "aws", lambda *args: {"builds": [record]})
    assert installer.reusable_image_build() is None
    record["buildStatus"] = "FAILED"
    assert installer.reusable_image_build() is None
    record["buildStatus"] = "SUCCEEDED"
    assert installer.reusable_image_build() == "old"


def test_destroy_includes_partial_foundation_without_saved_outputs(monkeypatch):
    from qps.hosted.operations import Operations

    operations = object.__new__(Operations)
    operations.configuration = {"billingStarted": 1, "cleanupPending": False}
    calls = []
    for name in ("janitor", "save", "cleanup_volumes", "close_credentials"):
        monkeypatch.setattr(operations, name, lambda name=name: calls.append(name))
    monkeypatch.setattr(operations, "tf", lambda *args: calls.append(args))
    operations.execute("destroy")
    assert ("foundation", "destroy") in calls
    assert operations.configuration["destroyed"]
    assert calls.index(("foundation", "destroy")) < calls.index("close_credentials")


def test_installation_waits_for_both_public_web_and_database_api(monkeypatch):
    import httpx

    installer = object.__new__(Installer)
    installer.installation = {"stack": {"url": "https://example.invalid"}}
    requests = []

    def response(request):
        requests.append(request.url.path)
        if request.url.path == "/":
            return httpx.Response(200, text="dashboard")
        return httpx.Response(503 if len(requests) == 2 else 200, json=[])

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(response), **kwargs),
    )
    monkeypatch.setattr("qps.hosted.install.time.sleep", lambda _: None)
    installer.wait_public()
    assert requests == ["/", "/api/v1/runs", "/", "/api/v1/runs"]


def test_credentials_are_restricted_even_when_replacing_existing_file(tmp_path):
    path = tmp_path / "owner.json"
    path.write_text("old")
    path.chmod(0o644)
    protected_json(path, {"password": "test-only"})
    assert path.stat().st_mode & 0o777 == 0o600
    assert "test-only" in read_key(path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        read_key(path)


def test_independent_cleanup_requires_committed_ownership_and_exact_provider_identity(monkeypatch):
    module = janitor()
    monkeypatch.setenv("PREFIX", "qps-test")
    calls = []

    def request(method, path, token):
        calls.append((method, path))
        return None

    monkeypatch.setattr(module, "request", request)
    records = [
        {
            "id": "alloc-one",
            "pod_id": "owned",
            "deadline": 50,
            "lease_until": 50,
            "state": "running",
            "profile": "runpod-qwen3-small",
        }
    ]
    pods = [
        {"id": "owned", "name": "qps-test-alloc-one"},
        {"id": "foreign", "name": "qps-test-unrecorded"},
    ]
    result = module.cleanup(records, pods, "test-token", 100)
    assert result == [{"allocationId": "alloc-one", "podId": "owned", "absent": True}]
    assert calls == [("DELETE", "/pods/owned"), ("GET", "/pods/owned")]
    calls.clear()
    pods[0]["id"] = "wrong-identity"
    assert module.cleanup(records, pods, "test-token", 100)[0]["identityConflict"]
    assert calls == []


def test_janitor_preserves_unexpired_and_cpu_allocations(monkeypatch):
    module = janitor()
    monkeypatch.setenv("PREFIX", "qps-test")
    monkeypatch.setattr(module, "request", lambda *args: pytest.fail("No provider calls allowed"))
    record = {
        "id": "alloc-one",
        "deadline": 200,
        "lease_until": 200,
        "state": "running",
        "profile": "runpod-qwen3-small",
    }
    assert module.cleanup([record], [{"name": "qps-test-alloc-one", "id": "pod"}], "", 100) == []
    record["profile"] = "cpu-fixture"
    assert module.cleanup([record], [], "", 100, force=True) == []

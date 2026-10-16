from types import SimpleNamespace

import httpx

from qps.hosted.forge import setup_project


def test_private_team_policy_is_preserved_and_embedding_is_not_advertised(monkeypatch):
    import wandb_workspaces.reports.v2 as wr

    calls = []
    client_type = httpx.Client

    def handle(request):
        import json

        body = json.loads(request.content)
        calls.append(body)
        if "QPSPolicy" in body["query"]:
            return httpx.Response(
                200, json={"data": {"entity": {"privateOnly": True, "defaultAccess": "PRIVATE"}}}
            )
        assert "upsertModel" in body["query"]
        assert "access" not in body["variables"]["input"]
        return httpx.Response(
            200,
            json={
                "data": {"upsertModel": {"project": {"name": "qps-platform", "access": "PRIVATE"}}}
            },
        )

    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_BASE_URL", "https://api.forge.coreweave.com")
    monkeypatch.setattr(
        "qps.hosted.forge.httpx.Client",
        lambda **kw: client_type(transport=httpx.MockTransport(handle), **kw),
    )
    monkeypatch.setattr(
        wr.Report, "from_url", lambda _: SimpleNamespace(url="https://example.invalid/report")
    )
    result = setup_project(
        "enfinity7b", "qps-platform", "test-only", report_url="https://example.invalid/report"
    )
    assert result["public"] is False
    assert result["visibilityReason"] == "Team privateOnly policy"
    assert len(calls) == 2


def test_public_project_uses_forge_user_read_without_public_writes(monkeypatch):
    import json

    import wandb_workspaces.reports.v2 as wr

    original = httpx.Client
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        if "QPSPolicy" in body["query"]:
            return httpx.Response(
                200, json={"data": {"entity": {"privateOnly": False, "defaultAccess": "PRIVATE"}}}
            )
        assert body["variables"]["input"]["access"] == "USER_READ"
        assert body["variables"]["input"]["name"] == "qps-platform"
        assert body["variables"]["input"]["entityName"] == "enfinity7b"
        return httpx.Response(
            200,
            json={
                "data": {
                    "upsertModel": {"project": {"name": "qps-platform", "access": "USER_READ"}}
                }
            },
        )

    monkeypatch.setattr(
        "qps.hosted.forge.httpx.Client",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    monkeypatch.setattr(
        wr.Report, "from_url", lambda _: SimpleNamespace(url="https://example.invalid/report")
    )
    result = setup_project(
        "enfinity7b", "qps-platform", "test-only", report_url="https://example.invalid/report"
    )
    assert result["public"] is True and len(calls) == 2
    assert not any("updateEntity" in c["query"] for c in calls)

"""Credential-free regression probe, also executed inside an immutable API image."""

from types import SimpleNamespace
from unittest.mock import patch

import wandb
import wandb_workspaces.reports.v2 as wr
from qps.hosted.forge import delete_experiment


def probe():
    deleted = []
    owner = "run-owned-experiment"
    api = SimpleNamespace(
        runs=lambda **kw: [],
        reports=lambda **kw: [
            SimpleNamespace(display_name=owner, url="https://example.invalid/owned"),
            SimpleNamespace(
                display_name="another-experiment", url="https://example.invalid/foreign"
            ),
        ],
    )
    settings = SimpleNamespace(forge_entity="enfinity7b", forge_project="qps-platform")
    with (
        patch("qps.hosted.settings.Settings.from_env", return_value=settings),
        patch.object(wandb, "Api", return_value=api),
        patch.object(
            wr.Report,
            "from_url",
            side_effect=lambda url: SimpleNamespace(
                delete=lambda: deleted.append(url) or True
            ),
        ),
    ):
        delete_experiment(
            {
                "resourceId": owner,
                "trackingAttempts": [owner + "-a1"],
                "reportUrl": None,
            }
        )
    assert deleted == ["https://example.invalid/owned"]
    return {
        "passed": True,
        "reportCreatedAfterPreviewRemoved": True,
        "foreignReportPreserved": True,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(probe()))

import json
from pathlib import Path

import pytest

from qps.artifacts import validate_manifest
from qps.contracts import HostedSubmission, recipe_digest
from qps.runtime import ArtifactRuntime
from qps.specs import RunSpec


def test_portable_checkpoint_identity_ignores_placement_but_not_recipe(local_run):
    _, _, spec = local_run
    relocated = spec.model_copy(deep=True)
    relocated.spec.artifacts.root = Path("/replacement/artifacts")
    relocated.spec.model.id = "/replacement/parent"
    relocated.spec.dataset.id = "/replacement/data.jsonl"
    assert recipe_digest(spec) == recipe_digest(relocated)
    relocated.spec.training.learning_rate *= 2
    assert recipe_digest(spec) != recipe_digest(relocated)


def test_hosted_profile_rejects_wrong_device_and_excess_runtime(local_run):
    _, _, spec = local_run
    with pytest.raises(ValueError, match="disagree"):
        HostedSubmission(run=spec, idempotency_key="request-1234")
    spec.spec.execution.max_wall_time_seconds = 2700
    accepted = HostedSubmission(
        run=spec, compute_profile="cpu-fixture", idempotency_key="request-1234"
    )
    assert accepted.run == spec
    spec.spec.execution.max_wall_time_seconds = 2701
    with pytest.raises(ValueError, match="2700"):
        HostedSubmission(run=spec, compute_profile="cpu-fixture", idempotency_key="request-1234")


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../b", "a//b", "a\\b", "."])
def test_remote_manifest_rejects_unsafe_paths(path):
    with pytest.raises(ValueError, match="Unsafe"):
        validate_manifest(
            {
                "version": 1,
                "id": "art-" + "a" * 32,
                "run_id": "run-" + "b" * 32,
                "kind": "merged_model",
                "metadata": {},
                "files": {path: {"sha256": "c" * 64, "size": 1}},
            }
        )


def test_runtime_does_not_need_original_database(successful):
    store, report = successful
    run = store.run(report["run_id"])
    artifact = store.artifact(store.output(run["id"], "export")["model"])
    runtime = ArtifactRuntime(
        Path(artifact["path"]), artifact["manifest_sha"], RunSpec.model_validate_json(run["spec"])
    )
    assert runtime.artifact(artifact["id"])["manifest_sha"] == artifact["manifest_sha"]
    assert json.loads(runtime.run(run["id"])["spec"])["metadata"]["name"] == run["name"]

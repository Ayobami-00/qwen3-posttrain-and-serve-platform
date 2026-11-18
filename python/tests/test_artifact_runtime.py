"""Standalone artifact metadata does not depend on a live training registry.

The fixture payloads test the artifact boundary, not model loading or generation.
"""

import json
import shutil
from pathlib import Path

import pytest

from qps.artifacts import Artifacts
from qps.runtime import ArtifactRuntime


def publish(local_run, tmp_path, kind="merged_model"):
    store, run, spec = local_run
    source = tmp_path / "payload"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "weights.bin").write_bytes(b"fixture bytes")
    identity = Artifacts(store, spec.spec.artifacts.root).publish(run["id"], kind, source, {})
    return store.artifact(identity)


def test_relocated_bundle_loads_after_the_source_registry_and_bundle_are_removed(
    local_run, tmp_path
):
    store, run, spec = local_run
    record = publish(local_run, tmp_path)
    destination = tmp_path / "relocated"
    shutil.copytree(record["path"], destination)
    shutil.rmtree(store.root)
    shutil.rmtree(record["path"])

    runtime = ArtifactRuntime(destination, record["manifest_sha"], spec)

    assert runtime.artifact(record["id"])["path"] == str(destination.resolve())
    assert runtime.artifact(record["id"])["run_id"] == run["id"]
    assert json.loads(runtime.run(run["id"])["spec"]) == spec.model_dump(mode="json", by_alias=True)


@pytest.mark.parametrize("change", ["manifest", "payload"])
def test_standalone_loading_rejects_modified_manifest_or_payload(local_run, tmp_path, change):
    _, _, spec = local_run
    record = publish(local_run, tmp_path)
    directory = Path(record["path"])
    if change == "manifest":
        manifest = directory / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        expected = "manifest has been modified"
    else:
        (directory / "files" / "weights.bin").write_bytes(b"changed bytes")
        expected = "checksum verification"

    with pytest.raises(ValueError, match=expected):
        ArtifactRuntime(directory, record["manifest_sha"], spec)


@pytest.mark.parametrize("kind", ["base_model", "lora_adapter", "training_checkpoint", "dataset"])
def test_standalone_serving_requires_a_merged_model(local_run, tmp_path, kind):
    _, _, spec = local_run
    record = publish(local_run, tmp_path, kind)
    with pytest.raises(ValueError, match="Serving requires a merged model artifact"):
        ArtifactRuntime(Path(record["path"]), record["manifest_sha"], spec)


def test_standalone_runtime_does_not_resolve_unrelated_identities(local_run, tmp_path):
    _, _, spec = local_run
    record = publish(local_run, tmp_path)
    runtime = ArtifactRuntime(Path(record["path"]), record["manifest_sha"], spec)

    with pytest.raises(ValueError, match="Unknown artifact"):
        runtime.artifact("art-" + "0" * 32)
    with pytest.raises(ValueError, match="Unknown artifact parent run"):
        runtime.run("run-" + "0" * 32)

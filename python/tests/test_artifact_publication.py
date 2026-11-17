"""Local publication must not register partial or corrupt filesystem outputs."""

from pathlib import Path

import pytest

from qps.artifacts import Artifacts


def source_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"original")
    return source


def test_interrupted_copy_removes_staging_without_a_registry_record(
    local_run, tmp_path, monkeypatch
):
    store, run, spec = local_run
    source = source_directory(tmp_path)
    publisher = Artifacts(store, spec.spec.artifacts.root)

    def interrupted_copy(source_path, destination):
        destination.mkdir(parents=True)
        (destination / "partial.bin").write_bytes(b"partial")
        raise OSError("simulated disk full")

    monkeypatch.setattr("qps.artifacts.shutil.copytree", interrupted_copy)
    with pytest.raises(OSError, match="simulated disk full"):
        publisher.publish(run["id"], "merged_model", source, {})

    assert store.artifacts(run["id"]) == []
    assert list(publisher.root.iterdir()) == []
    assert (source / "weights.bin").read_bytes() == b"original"


def test_reconciliation_refuses_corrupted_unregistered_outputs(local_run, tmp_path, monkeypatch):
    store, run, spec = local_run
    publisher = Artifacts(store, spec.spec.artifacts.root)

    def unavailable(*args):
        raise RuntimeError("registry unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(publisher, "_register", unavailable)
        with pytest.raises(RuntimeError, match="registry unavailable"):
            publisher.publish(run["id"], "merged_model", source_directory(tmp_path), {})

    final = next(publisher.root.glob("art-*"))
    (final / "files" / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum verification"):
        publisher.reconcile()
    assert store.artifacts(run["id"]) == []


def test_repeated_reconciliation_keeps_one_immutable_registry_record(local_run, tmp_path):
    store, run, spec = local_run
    publisher = Artifacts(store, spec.spec.artifacts.root)
    artifact_id = publisher.publish(run["id"], "merged_model", source_directory(tmp_path), {})
    record = store.artifact(artifact_id)

    publisher.reconcile()
    publisher.reconcile()

    assert store.artifacts(run["id"]) == [record]
    assert Path(record["path"]).is_dir()

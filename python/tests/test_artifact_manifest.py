"""Reject unsafe artifact envelopes before their paths reach filesystem or S3 joins."""

from hashlib import sha256

import pytest

from qps.artifacts import validate_manifest


@pytest.fixture
def manifest():
    return {
        "version": 1,
        "id": "art-" + "a" * 32,
        "run_id": "run-" + "b" * 32,
        "kind": "merged_model",
        "metadata": {},
        "files": {"weights/model.bin": {"size": 4, "sha256": sha256(b"data").hexdigest()}},
    }


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "../model.bin",
        "weights/../../model.bin",
        "/model.bin",
        "./model.bin",
        "weights//model.bin",
        "weights/",
        "weights\\model.bin",
    ],
)
def test_manifest_rejects_unsafe_or_noncanonical_file_paths(manifest, name):
    manifest["files"] = {name: next(iter(manifest["files"].values()))}
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        validate_manifest(manifest)


@pytest.mark.parametrize("size", [-1, True, False, 1.5, "4", None])
def test_file_sizes_require_nonnegative_integers_without_boolean_coercion(manifest, size):
    manifest["files"]["weights/model.bin"]["size"] = size
    with pytest.raises(ValueError, match="checksum or size"):
        validate_manifest(manifest)


@pytest.mark.parametrize("checksum", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, None])
def test_file_checksums_require_lowercase_sha256(manifest, checksum):
    manifest["files"]["weights/model.bin"]["sha256"] = checksum
    with pytest.raises(ValueError, match="checksum or size"):
        validate_manifest(manifest)


@pytest.mark.parametrize("files", [{}, [], None])
def test_file_inventory_must_be_a_nonempty_mapping(manifest, files):
    manifest["files"] = files
    with pytest.raises(ValueError, match="Invalid artifact inventory"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "run-" + "a" * 32),
        ("id", "art-" + "A" * 32),
        ("run_id", "run-" + "b" * 31),
        ("run_id", None),
    ],
)
def test_artifact_and_run_identities_cannot_be_confused(manifest, field, value):
    manifest[field] = value
    with pytest.raises(ValueError, match="Invalid artifact identity"):
        validate_manifest(manifest)


def test_nested_files_and_empty_file_checksums_are_valid_inventory_entries(manifest):
    manifest["files"]["metadata/empty.txt"] = {"size": 0, "sha256": sha256(b"").hexdigest()}
    validate_manifest(manifest)

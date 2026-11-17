"""Byte verification rejects changed inventories and symbolic file aliases."""

import pytest

from qps.artifacts import file_manifest, verify_directory


@pytest.fixture
def files(tmp_path):
    directory = tmp_path / "files"
    directory.mkdir()
    (directory / "weights.bin").write_bytes(b"weights-A")
    (directory / "tokenizer.json").write_text("{}")
    return directory, {"files": file_manifest(directory)}


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_verification_requires_the_exact_registered_inventory(files, change):
    directory, manifest = files
    if change == "missing":
        (directory / "weights.bin").unlink()
    else:
        (directory / "unregistered.bin").write_bytes(b"extra")

    with pytest.raises(ValueError, match="inventory differs"):
        verify_directory(directory, manifest)


def test_same_size_weight_corruption_is_detected_by_content_hash(files):
    directory, manifest = files
    path = directory / "weights.bin"
    original_size = path.stat().st_size
    path.write_bytes(b"weights-B")
    assert path.stat().st_size == original_size

    with pytest.raises(ValueError, match="checksum verification failed: weights.bin"):
        verify_directory(directory, manifest)


@pytest.mark.parametrize("scope", ["inside", "outside"])
def test_symbolic_files_are_rejected_even_when_the_target_exists(files, tmp_path, scope):
    directory, manifest = files
    path = directory / "weights.bin"
    target = directory / "tokenizer.json" if scope == "inside" else tmp_path / "outside.bin"
    if scope == "outside":
        target.write_bytes(path.read_bytes())
    expected_target = target.read_bytes()
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(ValueError, match="Unsafe artifact path|checksum verification failed"):
        verify_directory(directory, manifest)
    assert target.read_bytes() == expected_target


def test_file_inventory_preserves_nested_empty_files(tmp_path):
    directory = tmp_path / "files"
    (directory / "nested").mkdir(parents=True)
    (directory / "nested" / "empty.txt").write_bytes(b"")
    manifest = {"files": file_manifest(directory)}

    assert manifest["files"]["nested/empty.txt"]["size"] == 0
    verify_directory(directory, manifest)


def test_empty_source_directory_cannot_be_published_as_an_inventory(tmp_path):
    with pytest.raises(ValueError, match="Cannot publish an empty artifact"):
        file_manifest(tmp_path)

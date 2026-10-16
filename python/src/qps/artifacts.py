"""Checksummed immutable artifacts with filesystem-before-database commitment."""

import json
import re
import shutil
import uuid
from pathlib import Path

from qps.store import Store
from qps.util import atomic_json, digest, now


def validate_manifest(manifest: dict):
    """Validate untrusted envelopes before joining paths or allocating storage."""
    if manifest.get("version") != 1:
        raise ValueError("Unsupported artifact manifest version")
    for key, prefix in (("id", "art"), ("run_id", "run")):
        if not re.fullmatch(prefix + r"-[a-f0-9]{32}", str(manifest.get(key, ""))):
            raise ValueError("Invalid artifact identity")
    if not isinstance(manifest.get("metadata"), dict) or not isinstance(manifest.get("kind"), str):
        raise ValueError("Invalid artifact metadata")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > 10000:
        raise ValueError("Invalid artifact inventory")
    for name, info in files.items():
        path = Path(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or name == "."
        ):
            raise ValueError("Unsafe artifact path")
        if (
            not isinstance(info, dict)
            or type(info.get("size")) is not int
            or info["size"] < 0
            or not re.fullmatch(r"[a-f0-9]{64}", str(info.get("sha256", "")))
        ):
            raise ValueError("Invalid artifact file checksum or size")


def file_manifest(directory: Path) -> dict:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = {
                "sha256": digest(path),
                "size": path.stat().st_size,
            }
    if not result:
        raise ValueError("Cannot publish an empty artifact")
    return result


def verify_directory(directory: Path, manifest: dict):
    expected = manifest["files"]
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(expected):
        raise ValueError("Artifact file inventory differs from its manifest")
    for name, info in expected.items():
        path = directory / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unsafe artifact path")
        if (
            path.is_symlink()
            or digest(path) != info["sha256"]
            or path.stat().st_size != info["size"]
        ):
            raise ValueError(f"Artifact checksum verification failed: {name}")


class Artifacts:
    def __init__(self, store: Store, root: Path):
        self.store = store
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, run_id: str, kind: str, source: Path, metadata: dict) -> str:
        artifact_id = "art-" + uuid.uuid4().hex
        temporary = self.root / (".staging-" + artifact_id)
        final = self.root / artifact_id
        try:
            shutil.copytree(source, temporary / "files")
            manifest = {
                "version": 1,
                "id": artifact_id,
                "run_id": run_id,
                "kind": kind,
                "created": now(),
                "metadata": metadata,
                "files": file_manifest(temporary / "files"),
            }
            verify_directory(temporary / "files", manifest)
            atomic_json(temporary / "manifest.json", manifest)
            temporary.rename(final)
            self._register(final, manifest)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return artifact_id

    def _register(self, directory: Path, manifest: dict):
        with self.store.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?,?)",
                (
                    manifest["id"],
                    manifest["run_id"],
                    manifest["kind"],
                    str(directory),
                    digest(directory / "manifest.json"),
                    manifest["created"],
                ),
            )

    def reconcile(self):
        for path in self.root.glob("art-*/manifest.json"):
            with self.store.connect() as db:
                if db.execute(
                    "SELECT 1 FROM artifacts WHERE path=?", (str(path.parent),)
                ).fetchone():
                    continue
            manifest = json.loads(path.read_text())
            if manifest["id"] != path.parent.name:
                raise ValueError("Artifact directory identity mismatch")
            self.store.run(manifest["run_id"])
            verify_directory(path.parent / "files", manifest)
            self._register(path.parent, manifest)


def verify_artifact(store: Store, artifact_id: str) -> tuple[Path, dict]:
    record = store.artifact(artifact_id)
    directory = Path(record["path"])
    if digest(directory / "manifest.json") != record["manifest_sha"]:
        raise ValueError("Artifact manifest has been modified")
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["id"], manifest["run_id"], manifest["kind"]) != (
        artifact_id,
        record["run_id"],
        record["kind"],
    ):
        raise ValueError("Artifact identity mismatch")
    verify_directory(directory / "files", manifest)
    return directory / "files", manifest

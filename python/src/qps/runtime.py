"""A serving artifact can be loaded without the training process's SQLite registry."""

import json
from pathlib import Path

from qps.artifacts import validate_manifest, verify_directory
from qps.specs import RunSpec
from qps.util import canonical, digest


class ArtifactRuntime:
    def __init__(self, directory: Path, expected_sha: str, spec: RunSpec):
        self.directory = directory.resolve()
        if digest(self.directory / "manifest.json") != expected_sha:
            raise ValueError("Artifact manifest has been modified")
        manifest = json.loads((self.directory / "manifest.json").read_text())
        validate_manifest(manifest)
        if manifest["kind"] != "merged_model":
            raise ValueError("Serving requires a merged model artifact")
        verify_directory(self.directory / "files", manifest)
        self.manifest = manifest
        self.spec = spec
        self.sha = expected_sha

    def artifact(self, artifact_id):
        if artifact_id != self.manifest["id"]:
            raise ValueError("Unknown artifact")
        return {
            "id": artifact_id,
            "run_id": self.manifest["run_id"],
            "kind": self.manifest["kind"],
            "path": str(self.directory),
            "manifest_sha": self.sha,
        }

    def run(self, run_id):
        if run_id != self.manifest["run_id"]:
            raise ValueError("Unknown artifact parent run")
        return {"id": run_id, "spec": canonical(self.spec.model_dump(mode="json", by_alias=True))}

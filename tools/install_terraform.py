"""Checksum-verified Terraform download for managed Linux operations jobs."""

import hashlib
import io
import platform
import zipfile
from pathlib import Path

import httpx

architecture = "arm64" if platform.machine() in {"arm64", "aarch64"} else "amd64"
name = f"terraform_1.16.6_linux_{architecture}.zip"
base = "https://releases.hashicorp.com/terraform/1.16.6/"
with httpx.Client(timeout=60) as client:
    checks = client.get(base + "terraform_1.16.6_SHA256SUMS")
    checks.raise_for_status()
    archive = client.get(base + name)
    archive.raise_for_status()
expected = next(
    line.split()[0] for line in checks.text.splitlines() if line.endswith(name)
)
assert hashlib.sha256(archive.content).hexdigest() == expected, (
    "Terraform checksum mismatch"
)
path = Path(".cache/tools/terraform")
path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(archive.content)) as package:
    path.write_bytes(package.read("terraform"))
path.chmod(0o755)

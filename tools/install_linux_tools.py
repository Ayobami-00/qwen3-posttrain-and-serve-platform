"""Pinned/checksummed kubectl and Helm for disposable Linux integration jobs."""

import hashlib
import io
import tarfile
from pathlib import Path

import httpx

directory = Path(".cache/tools")
directory.mkdir(parents=True, exist_ok=True)
with httpx.Client(timeout=90, follow_redirects=True) as client:
    for name, url, checksum_url, member in [
        (
            "kubectl",
            "https://dl.k8s.io/release/v1.36.5/bin/linux/amd64/kubectl",
            "https://dl.k8s.io/release/v1.36.5/bin/linux/amd64/kubectl.sha256",
            None,
        ),
        (
            "helm",
            "https://get.helm.sh/helm-v4.3.1-linux-amd64.tar.gz",
            "https://get.helm.sh/helm-v4.3.1-linux-amd64.tar.gz.sha256sum",
            "linux-amd64/helm",
        ),
    ]:
        response = client.get(url)
        response.raise_for_status()
        checksum = client.get(checksum_url)
        checksum.raise_for_status()
        assert (
            hashlib.sha256(response.content).hexdigest() == checksum.text.split()[0]
        ), name + " checksum mismatch"
        content = response.content
        if member:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
                content = archive.extractfile(member).read()
        path = directory / name
        path.write_bytes(content)
        path.chmod(0o755)

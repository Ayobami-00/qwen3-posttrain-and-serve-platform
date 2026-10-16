"""Build a Forge-only API fix while preserving verified, unchanged runtime images.

Two immutable Git archives must agree byte-for-byte on all other runtime inputs.
The previous Linux gate is retained explicitly; the new API runs its Forge probe.
"""

import io
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

import boto3


def runtime_input(name):
    return (
        name.startswith(("python/src/", "operator/", "web/", "contracts/"))
        or name in {"python/pyproject.toml", "python/uv.lock"}
        or (name.startswith("containers/") and name.endswith("Dockerfile"))
    ) and not (
        name in {"python/src/qps/hosted/forge.py", "python/src/qps/hosted/api.py"}
        or name.startswith("web/tests/")
        or name.endswith("_test.go")
    )


def archive_inputs(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/") and runtime_input(name)
        }


def main():
    base, sha = os.environ["QPS_API_BASE_SHA"], os.environ["QPS_SOURCE_SHA"]
    if not all(re.fullmatch(r"[a-f0-9]{40}", value) for value in (base, sha)):
        raise ValueError("API patch requires immutable source revisions")
    region, account, prefix = (
        os.environ[n] for n in ("QPS_REGION", "QPS_ACCOUNT_ID", "QPS_PREFIX")
    )
    s3 = boto3.client("s3", region_name=region)
    reports = os.environ["QPS_REPORTS_BUCKET"]
    sources = f"{prefix}-{account}-{region}-sources"

    def read(bucket, key):
        response = s3.get_object(Bucket=bucket, Key=key)
        if response["ContentLength"] > 20 * 1024**2:
            raise ValueError("Unexpected build metadata/archive size")
        try:
            return response["Body"].read()
        finally:
            response["Body"].close()

    prior = json.loads(read(reports, f"builds/{base}/build.json"))
    if prior["sourceSha"] != base or not prior["linuxCpuKubernetesPassed"]:
        raise ValueError("The inherited Linux integration gate was not verified")
    if archive_inputs(read(sources, f"sources/{base}.zip")) != archive_inputs(
        read(sources, f"sources/{sha}.zip")
    ):
        raise ValueError("A non-Forge runtime changed; use the complete image build")
    images = json.loads(read(reports, f"builds/{base}/images.json"))
    tag = f"{account}.dkr.ecr.{region}.amazonaws.com/{prefix}/api:{sha}"
    subprocess.run(
        [
            "docker",
            "build",
            "--platform=linux/amd64",
            "--build-arg",
            "QPS_SOURCE_SHA=" + sha,
            "-f",
            "containers/python.Dockerfile",
            "--target",
            "api",
            "-t",
            tag,
            ".",
        ],
        check=True,
    )
    subprocess.run(
        ["docker", "run", "--rm", "-i", "--entrypoint", "python", tag, "-"],
        input=Path("tools/forge_cleanup_probe.py").read_bytes(),
        check=True,
    )
    subprocess.run(["docker", "push", tag], check=True)
    detail = boto3.client("ecr", region_name=region).describe_images(
        repositoryName=f"{prefix}/api", imageIds=[{"imageTag": sha}]
    )["imageDetails"][0]
    images["api"] = tag.rsplit(":", 1)[0] + "@" + detail["imageDigest"]
    sizes = {**prior["imageSizes"], "api": detail["imageSizeInBytes"]}
    lineage = dict(prior.get("runtimeSources", {name: base for name in images}))
    lineage["api"] = sha
    for name, value in {
        "images.json": images,
        "build.json": {
            "sourceSha": sha,
            "linuxCpuKubernetesPassed": True,
            "inheritedLinuxCpuKubernetesSha": prior.get(
                "inheritedLinuxCpuKubernetesSha", base
            ),
            "runtimeSources": lineage,
            "forgeApiProbePassed": True,
            "otherRuntimeInputsIdentical": True,
            "imageSizes": sizes,
        },
    }.items():
        s3.put_object(
            Bucket=reports, Key=f"builds/{sha}/{name}", Body=json.dumps(value).encode()
        )


if __name__ == "__main__":
    main()

"""Linux image build and CPU Kubernetes acceptance, before opening the EKS window."""

import json
import os
import subprocess
import time
from pathlib import Path

import boto3


def call(args, **kwargs):
    print("Build task: " + " ".join(args[:4]), flush=True)
    return subprocess.run(args, check=True, **kwargs)


def main():
    account = os.environ["QPS_ACCOUNT_ID"]
    region = os.environ["QPS_REGION"]
    prefix = os.environ["QPS_PREFIX"]
    sha = os.environ["QPS_SOURCE_SHA"]
    if boto3.client("sts").get_caller_identity()["Account"] != account:
        raise ValueError("Build account guard failed")
    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"
    password = subprocess.check_output(
        ["aws", "ecr", "get-login-password", "--region", region]
    )
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", registry],
        input=password,
        check=True,
    )
    if os.environ.get("QPS_API_BASE_SHA"):
        from cloud_api_patch import main as build_api_patch

        build_api_patch()
        return
    tags = {}
    definitions = [
        ("api", "containers/python.Dockerfile", "api"),
        ("worker-cpu", "containers/python.Dockerfile", "cpu-worker"),
        ("operator", "containers/operator.Dockerfile", None),
        ("web", "containers/web.Dockerfile", None),
    ]
    started = time.time()
    for name, path, target in definitions:
        tag = f"{registry}/{prefix}/{name}:{sha}"
        args = [
            "docker",
            "build",
            "--platform=linux/amd64",
            "--build-arg",
            "QPS_SOURCE_SHA=" + sha,
            "-f",
            path,
            "-t",
            tag,
        ]
        if target:
            args += ["--target", target]
        call(args + ["."])
        tags[name] = tag
    Path(".qps").mkdir(exist_ok=True)
    Path(".qps/build-tags.json").write_text(json.dumps(tags))
    # No Runpod API key or AWS administrator credentials enter test pods.
    call(["python/.venv/bin/python", "tools/kubernetes_acceptance.py"])
    if time.time() - started > 1800:
        raise RuntimeError(
            "CPU build exceeded the reserved build allowance; CUDA build not started"
        )
    # Large CUDA dependencies and real weights are never built/downloaded on the Mac.
    worker = f"{registry}/{prefix}/worker:{sha}"
    call(
        [
            "docker",
            "build",
            "--platform=linux/amd64",
            "--build-arg",
            "QPS_SOURCE_SHA=" + sha,
            "-f",
            "containers/worker.Dockerfile",
            "-t",
            worker,
            ".",
        ]
    )
    tags["worker"] = worker
    ecr = boto3.client("ecr", region_name=region)
    images = {}
    sizes = {}
    for name, tag in tags.items():
        existing = ecr.describe_images(
            repositoryName=f"{prefix}/{name}", filter={"tagStatus": "TAGGED"}
        )["imageDetails"]
        match = next((x for x in existing if sha in x.get("imageTags", [])), None)
        if not match:
            call(["docker", "push", tag])
            match = ecr.describe_images(
                repositoryName=f"{prefix}/{name}", imageIds=[{"imageTag": sha}]
            )["imageDetails"][0]
        images[name] = tag.rsplit(":", 1)[0] + "@" + match["imageDigest"]
        sizes[name] = match["imageSizeInBytes"]
    s3 = boto3.client("s3", region_name=region)
    for name, body in {
        "images.json": images,
        "build.json": {
            "sourceSha": sha,
            "elapsedSeconds": time.time() - started,
            "linuxCpuKubernetesPassed": True,
            "imageSizes": sizes,
        },
    }.items():
        s3.put_object(
            Bucket=os.environ["QPS_REPORTS_BUCKET"],
            Key=f"builds/{sha}/{name}",
            Body=json.dumps(body).encode(),
        )
    for path in Path(".qps").glob("dashboard-*.png"):
        s3.upload_file(
            str(path), os.environ["QPS_REPORTS_BUCKET"], f"builds/{sha}/{path.name}"
        )


if __name__ == "__main__":
    main()

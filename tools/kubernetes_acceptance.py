"""Disposable Linux-only Kind proof with PostgreSQL, S3 emulator and real Go controller."""

import json
import subprocess
import time
from pathlib import Path

import httpx
import yaml

NODE_IMAGE = "kindest/node:v1.35.0@sha256:452d707d4862f52530247495d180205e029056831160e22870e37e3f6c1ac31f"


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def wait(url, seconds=120):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("Kubernetes endpoint did not become healthy")


def fixture(name, image, env, port):
    return [
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": name, "namespace": "qps"},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": name}},
                "template": {
                    "metadata": {"labels": {"app": name}},
                    "spec": {
                        "automountServiceAccountToken": False,
                        "containers": [
                            {
                                "name": name,
                                "image": image,
                                "env": [
                                    {"name": k, "value": v} for k, v in env.items()
                                ],
                                "ports": [{"containerPort": port}],
                                "resources": {
                                    "requests": {"cpu": "100m", "memory": "128Mi"},
                                    "limits": {"memory": "512Mi"},
                                },
                            }
                        ],
                    },
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": name, "namespace": "qps"},
            "spec": {
                "selector": {"app": name},
                "ports": [{"port": port, "targetPort": port}],
            },
        },
    ]


def main():
    tags = json.loads(Path(".qps/build-tags.json").read_text())
    processes = []
    run(
        "kind",
        "create",
        "cluster",
        "--name",
        "qps-acceptance",
        "--image",
        NODE_IMAGE,
        "--wait",
        "120s",
    )
    try:
        for image in tags.values():
            run("kind", "load", "docker-image", "--name", "qps-acceptance", image)
        run("kubectl", "create", "namespace", "qps")
        objects = fixture(
            "qps-postgres",
            "postgres:18.6",
            {
                "POSTGRES_USER": "qps",
                "POSTGRES_PASSWORD": "qps-kind-only",
                "POSTGRES_DB": "qps",
            },
            5432,
        )
        objects += fixture(
            "qps-moto", "motoserver/moto:5.2.4", {"MOTO_PORT": "5000"}, 5000
        )
        run("kubectl", "apply", "-f", "-", input=yaml.safe_dump_all(objects).encode())
        for name in ("qps-postgres", "qps-moto"):
            run(
                "kubectl",
                "-n",
                "qps",
                "rollout",
                "status",
                "deployment/" + name,
                "--timeout=180s",
            )
        p = subprocess.Popen(
            ["kubectl", "-n", "qps", "port-forward", "service/qps-moto", "15000:5000"],
            stdout=subprocess.DEVNULL,
        )
        processes.append(p)
        wait("http://127.0.0.1:15000/")
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url="http://127.0.0.1:15000",
            region_name="eu-west-2",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        s3.create_bucket(
            Bucket="qps-development",
            CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
        )
        s3.put_bucket_versioning(
            Bucket="qps-development", VersioningConfiguration={"Status": "Enabled"}
        )
        values = {
            "images": tags,
            "monitoring": False,
            "apiMigration": True,
            "apiEnvironment": {
                "QPS_AUTH_MODE": "development",
                "QPS_PUBLIC_URL": "http://127.0.0.1:8080",
                "QPS_DATABASE_URL": "postgresql+psycopg://qps:qps-kind-only@qps-postgres:5432/qps",
                "QPS_DEV_TOKEN": "browser-test-owner-token",
                "QPS_INTERNAL_TOKEN": "browser-test-internal-token-32-characters",
                "QPS_WORKER_SIGNING_KEY": "browser-test-worker-signing-key-32-characters",
                "QPS_S3_ENDPOINT": "http://qps-moto:5000",
                "AWS_ACCESS_KEY_ID": "testing",
                "AWS_SECRET_ACCESS_KEY": "testing",
                "QPS_WORKER_API_URL": "http://qps-api:8080",
                "QPS_CPU_IMAGE": tags["worker-cpu"],
                "QPS_FORGE_ENABLED": "false",
            },
            "operatorEnvironment": {
                "QPS_INTERNAL_TOKEN": "browser-test-internal-token-32-characters"
            },
        }
        Path(".qps/kind-values.yaml").write_text(yaml.safe_dump(values))
        run(
            "helm",
            "upgrade",
            "--install",
            "qps",
            "deploy/helm/qps",
            "-n",
            "qps",
            "-f",
            ".qps/kind-values.yaml",
            "--wait",
            "--timeout",
            "5m",
        )
        for name, port in (("qps-api", 8080), ("qps-web", 3000)):
            processes.append(
                subprocess.Popen(
                    [
                        "kubectl",
                        "-n",
                        "qps",
                        "port-forward",
                        "service/" + name,
                        f"{port}:{port}",
                    ],
                    stdout=subprocess.DEVNULL,
                )
            )
        wait("http://127.0.0.1:8080/readyz")
        wait("http://127.0.0.1:3000/")
        run("npm", "--prefix", "web", "run", "test:e2e")
        print("Linux Kubernetes/browser acceptance passed", flush=True)
    except Exception:
        # Only dummy test credentials exist in this cluster. Do not reuse for live logs.
        run(
            "kubectl",
            "-n",
            "qps",
            "get",
            "pods,jobs,trainingruns,computeallocations,modeldeployments",
            "-o",
            "wide",
        )
        run("kubectl", "-n", "qps", "logs", "deployment/qps-api", "--tail=80")
        run("kubectl", "-n", "qps", "logs", "deployment/qps-operator", "--tail=80")
        raise
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=10)
        run("kind", "delete", "cluster", "--name", "qps-acceptance")


if __name__ == "__main__":
    main()

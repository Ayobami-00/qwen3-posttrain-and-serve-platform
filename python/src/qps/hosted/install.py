"""Reproducible installer. Credentials stay in the AWS SDK chain or protected files."""

import hashlib
import json
import os
import platform
import secrets
import shutil
import subprocess
import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import yaml
from pydantic import Field

from qps.specs import Contract


class PlatformConfig(Contract):
    account_id: str = Field(pattern=r"^\d{12}$")
    region: str = "eu-west-2"
    aws_profile: str = "qps-bootstrap"
    project: str = "qps"
    environment: str = "test"
    aws_total_usd: float = Field(default=30, gt=0, le=30)
    aws_forecast_ceiling_usd: float = Field(default=20, gt=0, le=20)
    lifetime_hours: int = Field(default=24, ge=1, le=24)
    runpod_total_usd: float = Field(default=3, gt=0, le=3)
    runpod_hourly_usd: float = Field(default=1, gt=0, le=1)
    validation_gpu_image_pulls: int = Field(default=5, ge=5, le=10)
    runpod_key_file: Path = Path("~/.config/qps/runpod-api-key")
    forge_key_file: Path = Path("~/.config/qps/wandb-api-key")
    forge_entity: str = "enfinity7b"
    forge_project: str = "qps-platform"


def forecast(hours=24, gpu_image_bytes=8 * 1024**3, ceiling=20, gpu_image_pulls=5):
    """Conservative London estimate. Revalidate unit rates before live admission."""
    model_transfer_gib = 8 + max(0, gpu_image_pulls - 5) * 3
    hourly = {
        "eks": 0.10,
        "cpu_nodes": 2 * 0.11655,
        "rds": 0.018,
        "nat": 0.052,
        "alb": 0.0252,
        "public_ipv4": 0.005,
    }
    fixed = {
        "builds": 1.50,
        "block_storage": 0.45,
        "nat_data": 0.70,
        "internet_transfer": (gpu_image_pulls * gpu_image_bytes / 1024**3 + model_transfer_gib)
        * 0.09,
        "private_retention": 0.30,
        "ecr": 0.40,
        "logs": 0.20,
        "cloudfront": 0.20,
        "secrets": 1.60,
        "cleanup_foundation": 0.20,
        "backups": 0.05,
        "load_balancer_capacity": 0.20,
    }
    import math

    requested = hours
    # Cold pulls cover two training workers, one replacement, serving and restart.
    affordable = (
        math.floor(((ceiling - 0.50) / 1.20 - sum(fixed.values())) / sum(hourly.values()) * 4) / 4
    )
    hours = min(hours, max(0, affordable))
    subtotal = sum(hourly.values()) * hours + sum(fixed.values())
    return {
        "currency": "USD",
        "hours": hours,
        "requestedHours": requested,
        "requestedLifetimeEstimate": round(
            (sum(hourly.values()) * requested + sum(fixed.values())) * 1.20, 2
        ),
        "coldGpuImageBytes": gpu_image_bytes,
        "coldGpuImagePulls": gpu_image_pulls,
        "modelTransferGiB": model_transfer_gib,
        "hourly": hourly,
        "fixed": fixed,
        "taxAndUncertaintyMultiplier": 1.20,
        "estimate": round(subtotal * 1.20, 2),
        "hardCap": False,
        "assumptions": f"London on-demand; no free tier, credits or cached image layers; {gpu_image_pulls} GPU image pulls plus {model_transfer_gib} GiB model downloads; 7/30-day retention; earlier budget cutoff permitted",
    }


def protected_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def read_key(path):
    path = path.expanduser().resolve()
    if path.stat().st_mode & 0o077:
        raise ValueError(f"Credential file must have mode 0600: {path}")
    value = path.read_text().strip()
    if not value:
        raise ValueError(f"Credential file is empty: {path}")
    return value


class Installer:
    def __init__(self, state_root, config=None):
        self.repo = Path(__file__).resolve().parents[4]
        self.directory = Path(state_root).resolve() / "platform" / "test"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config).resolve() if config else self.directory / "config.json"
        if self.config_path.exists():
            self.config = PlatformConfig.model_validate(
                yaml.safe_load(self.config_path.read_text())
            )
        else:
            raise ValueError("Provide --config with the platform installation configuration")
        if (self.config.region, self.config.project, self.config.environment) != (
            "eu-west-2",
            "qps",
            "test",
        ):
            raise ValueError("This validation installer supports qps/test in eu-west-2")
        protected_json(self.directory / "config.json", self.config.model_dump(mode="json"))
        self.env = os.environ.copy()
        self.env.update(
            AWS_PROFILE=self.config.aws_profile,
            AWS_REGION=self.config.region,
            AWS_DEFAULT_REGION=self.config.region,
            AWS_PAGER="",
        )
        self.install_path = self.directory / "installation.json"
        self.installation = (
            json.loads(self.install_path.read_text()) if self.install_path.exists() else {}
        )

    def command(self, args, cwd=None, capture=True, env=None):
        result = subprocess.run(
            args, cwd=cwd or self.repo, env=env or self.env, text=True, capture_output=capture
        )
        if result.returncode:
            # Command output can contain generated credentials. Save it privately, never echo it.
            if capture:
                protected_json(
                    self.directory / "command-error.json",
                    {"command": args[:3], "stdout": result.stdout, "stderr": result.stderr},
                )
            raise RuntimeError(
                f"{args[0]} failed; protected diagnostics: {self.directory / 'command-error.json'}"
            )
        return result.stdout if capture else ""

    def aws(self, *args):
        return json.loads(self.command(["aws", *args, "--output", "json"]) or "{}")

    def identity(self):
        identity = self.aws("sts", "get-caller-identity")
        if identity["Account"] != self.config.account_id:
            raise ValueError("AWS account does not match installation account guard")
        return {
            "account": identity["Account"],
            "principal": identity["Arn"],
            "region": self.config.region,
        }

    def terraform(self):
        binary = self.repo / ".cache/tools/terraform"
        if (
            binary.exists()
            and json.loads(self.command([str(binary), "version", "-json"]))["terraform_version"]
            == "1.16.6"
        ):
            return str(binary)
        architecture = "arm64" if platform.machine() in {"arm64", "aarch64"} else "amd64"
        osname = "darwin" if platform.system() == "Darwin" else "linux"
        archive = f"terraform_1.16.6_{osname}_{architecture}.zip"
        base = "https://releases.hashicorp.com/terraform/1.16.6/"
        with httpx.Client(timeout=60, trust_env=False) as client:
            checks = client.get(base + "terraform_1.16.6_SHA256SUMS")
            checks.raise_for_status()
            payload = client.get(base + archive)
            payload.raise_for_status()
        expected = next(
            line.split()[0] for line in checks.text.splitlines() if line.endswith(archive)
        )
        if hashlib.sha256(payload.content).hexdigest() != expected:
            raise ValueError("Terraform download checksum mismatch")
        import io

        binary.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload.content)) as package:
            binary.write_bytes(package.read("terraform"))
        binary.chmod(0o755)
        return str(binary)

    def sdk_bridge(self):
        # aws login refresh is implemented by AWS CLI; Terraform providers call credential_process.
        bridge = self.directory / "aws-config"
        profile = self.config.aws_profile
        if not profile.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Unsupported AWS profile name")
        bridge.write_text(
            f"[profile qps-terraform]\nregion = {self.config.region}\ncredential_process = aws configure export-credentials --profile {profile} --format process\n"
        )
        bridge.chmod(0o600)
        env = self.env.copy()
        env.update(AWS_PROFILE="qps-terraform", AWS_CONFIG_FILE=str(bridge))
        # Nested CLI must use the original config, avoiding a recursive credential process.
        wrapper = self.directory / "aws-credentials"
        original = os.environ.get("AWS_CONFIG_FILE", str(Path.home() / ".aws/config"))
        import shlex

        wrapper.write_text(
            "#!/bin/sh\nexec env AWS_CONFIG_FILE="
            + shlex.quote(original)
            + " aws configure export-credentials --profile "
            + shlex.quote(profile)
            + " --format process\n"
        )
        wrapper.chmod(0o700)
        bridge.write_text(
            f"[profile qps-terraform]\nregion = {self.config.region}\ncredential_process = {shlex.quote(str(wrapper))}\n"
        )
        return env

    def save(self):
        protected_json(self.install_path, self.installation)
        if self.installation.get("bootstrap"):
            self.command(
                [
                    "aws",
                    "s3",
                    "cp",
                    str(self.install_path),
                    "s3://"
                    + self.installation["bootstrap"]["buckets"]["reports"]
                    + "/installation/current.json",
                    "--only-show-errors",
                ]
            )

    def build(self, project, operation=None):
        args = ["codebuild", "start-build", "--project-name", project]
        if operation:
            args += [
                "--environment-variables-override",
                json.dumps([{"name": "QPS_OPERATION", "value": operation, "type": "PLAINTEXT"}]),
            ]
        build = self.aws(*args)["build"]
        self.installation.setdefault("builds", []).append(
            {"id": build["id"], "operation": operation or "images"}
        )
        self.save()
        print("Managed build started: " + build["id"], flush=True)
        return self.wait_build(build["id"])

    def wait_build(self, identity):
        last = ""
        while True:
            record = self.aws("codebuild", "batch-get-builds", "--ids", identity)["builds"][0]
            phase = record.get("currentPhase", "")
            if phase != last:
                print("Managed build: " + phase, flush=True)
                last = phase
            if record["buildStatus"] != "IN_PROGRESS":
                if record["buildStatus"] != "SUCCEEDED":
                    raise RuntimeError(
                        "Managed build failed: " + identity + "; inspect its scoped CloudWatch log"
                    )
                return record
            time.sleep(15)

    def image_inputs(self, revision):
        # The installer runs from the source archive on the host/build service. Its own
        # implementation and Helm/Terraform do not change the serving/training binaries.
        tree = self.command(
            [
                "git",
                "ls-tree",
                "-r",
                revision,
                "--",
                "containers",
                "python/src",
                "python/pyproject.toml",
                "python/uv.lock",
                "operator",
                "web",
                "contracts/container-bases.json",
            ]
        )
        lines = []
        for line in tree.splitlines():
            path = line.rsplit("\t", 1)[-1]
            if (
                path in {"python/src/qps/hosted/install.py", "python/src/qps/hosted/operations.py"}
                or path.startswith("web/tests/")
                or path.endswith("_test.go")
            ):
                continue
            lines.append(line)
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()

    def reusable_image_build(self):
        expected = self.image_inputs(self.installation["sourceSha"])
        previous = self.installation.get("imagesBuildSha")
        if previous and self.image_inputs(previous) == expected:
            return previous
        # A local coordinator may have stopped while the independent CodeBuild job
        # finished. Recover only a successful owned build, never an incomplete image.
        identities = [
            b["id"] for b in self.installation.get("builds", []) if b["operation"] == "images"
        ]
        if not identities:
            return None
        records = self.aws("codebuild", "batch-get-builds", "--ids", *identities[-100:])["builds"]
        for record in sorted(records, key=lambda r: r["startTime"], reverse=True):
            if record["buildStatus"] != "SUCCEEDED":
                continue
            values = {
                v["name"]: v.get("value") for v in record["environment"]["environmentVariables"]
            }
            revision = values.get("QPS_SOURCE_SHA")
            if revision and self.image_inputs(revision) == expected:
                return revision
        return None

    def wait_public(self, seconds=180):
        url = self.installation["stack"]["url"].rstrip("/")
        deadline = time.monotonic() + seconds
        with httpx.Client(timeout=10, trust_env=False) as client:
            while time.monotonic() < deadline:
                try:
                    web = client.get(url + "/")
                    api = client.get(url + "/api/v1/runs")
                    if web.status_code == api.status_code == 200 and isinstance(api.json(), list):
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                time.sleep(3)
        raise RuntimeError("Public dashboard/API readiness timed out; inspect the owned deployment")

    def refresh(self):
        bucket = self.installation["bootstrap"]["buckets"]["reports"]
        self.command(
            [
                "aws",
                "s3",
                "cp",
                f"s3://{bucket}/installation/current.json",
                str(self.install_path),
                "--only-show-errors",
            ]
        )
        self.installation = json.loads(self.install_path.read_text())

    def execute(self, operation):
        self.identity()
        if operation == "doctor":
            for file in (self.config.runpod_key_file, self.config.forge_key_file):
                read_key(file)
            return {
                "aws": self.identity(),
                "forecast": forecast(
                    self.config.lifetime_hours,
                    gpu_image_pulls=self.config.validation_gpu_image_pulls,
                ),
                "credentialFiles": "present with restricted permissions",
                "paidResourcesCreated": False,
            }
        if operation == "init":
            return self.initialize()
        if not self.installation.get("bootstrap"):
            raise ValueError("Run qps platform init first")
        if operation == "status":
            self.refresh()
            return {
                k: v
                for k, v in self.installation.items()
                if k in {"stack", "expiresAt", "budgetDeadline", "destroyed", "forecast", "builds"}
            }
        if operation == "plan":
            result = forecast(
                self.config.lifetime_hours, gpu_image_pulls=self.config.validation_gpu_image_pulls
            )
            if result["estimate"] >= self.config.aws_forecast_ceiling_usd:
                raise ValueError("Forecast is not below the deployment ceiling")
            self.installation["forecast"] = result
            self.save()
            self.build(self.installation["bootstrap"]["operations_project"], "plan")
            return result
        if operation == "apply":
            return self.apply()
        if operation in {"destroy", "backup", "restore"}:
            project = (
                self.installation["bootstrap"]["operations_project"]
                if operation == "destroy"
                else self.installation["stack"]["cluster_project"]
            )
            self.build(project, operation)
            self.refresh()
            return {"operation": operation, "destroyed": self.installation.get("destroyed", False)}
        raise ValueError("Unknown installation operation")

    def initialize(self):
        previous = dict(self.installation)
        if previous.get("destroyed") or (previous and time.time() >= previous["budgetDeadline"]):
            raise ValueError("Do not reset an expired installation's budget or lifetime")
        status = self.command(["git", "status", "--porcelain", "--untracked-files=normal"])
        if status:
            raise ValueError(
                "Commit the tested implementation before creating the immutable cloud source"
            )
        sha = self.command(["git", "rev-parse", "HEAD"]).strip()
        expires = (
            previous.get("expiresAt")
            or (datetime.now(UTC) + timedelta(hours=self.config.lifetime_hours)).isoformat()
        )
        variables = {
            "account_id": self.config.account_id,
            "region": self.config.region,
            "source_sha": sha,
            "source_key": f"sources/{sha}.zip",
            "expires_at": expires,
        }
        root = self.directory / "terraform/bootstrap"
        shutil.copytree(
            self.repo / "infra/terraform",
            root.parent,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".terraform", "*.tfstate*", "janitor.zip"),
        )
        shutil.copytree(self.repo / "infra/lambda", self.directory / "lambda", dirs_exist_ok=True)
        protected_json(root / "run.tfvars.json", variables)
        tf = self.terraform()
        env = self.sdk_bridge()
        self.command([tf, "init", "-input=false"], cwd=root, env=env)
        self.command(
            [tf, "apply", "-input=false", "-auto-approve", "-var-file=run.tfvars.json"],
            cwd=root,
            env=env,
        )
        outputs = json.loads(self.command([tf, "output", "-json"], cwd=root, env=env))
        bootstrap = {k: v["value"] for k, v in outputs.items()}
        self.installation = {
            **previous,
            "accountId": self.config.account_id,
            "region": self.config.region,
            "prefix": "qps-test",
            "sourceSha": sha,
            "sourceKey": variables["source_key"],
            "expiresAt": expires,
            "bootstrap": bootstrap,
            "created": previous.get("created", time.time()),
            "budgetDeadline": previous.get(
                "budgetDeadline", datetime.fromisoformat(expires).timestamp()
            ),
            "forecast": forecast(
                self.config.lifetime_hours, gpu_image_pulls=self.config.validation_gpu_image_pulls
            ),
            "destroyed": False,
        }
        bucket = bootstrap["buckets"]["state"]
        (root / "backend.tf").write_text('terraform {\n backend "s3" {}\n}\n')
        if not previous.get("bootstrap"):
            self.command(
                [
                    tf,
                    "init",
                    "-migrate-state",
                    "-force-copy",
                    "-input=false",
                    f"-backend-config=bucket={bucket}",
                    "-backend-config=key=test/bootstrap.tfstate",
                    "-backend-config=region=eu-west-2",
                    "-backend-config=use_lockfile=true",
                    "-backend-config=encrypt=true",
                ],
                cwd=root,
                env=env,
            )
        archive = self.directory / "source.zip"
        self.command(["git", "archive", "--format=zip", "--output=" + str(archive), sha])
        self.command(
            [
                "aws",
                "s3",
                "cp",
                str(archive),
                f"s3://{bootstrap['buckets']['sources']}/{variables['source_key']}",
                "--only-show-errors",
            ]
        )
        owner_file = self.directory / "owner-credentials.json"
        owner = (
            json.loads(owner_file.read_text())
            if owner_file.exists()
            else {"username": "owner", "password": "Qp7!" + secrets.token_urlsafe(28)}
        )
        protected_json(self.directory / "owner-credentials.json", owner)
        for name, value in {
            "owner": owner,
            "janitor": {"runpod_api_key": read_key(self.config.runpod_key_file)},
        }.items():
            self.put_secret(bootstrap["secrets"][name], value)
        self.save()
        from qps.hosted.forge import setup_project

        self.installation["tracking"] = setup_project(
            self.config.forge_entity,
            self.config.forge_project,
            read_key(self.config.forge_key_file),
            report_url=previous.get("tracking", {}).get("reportUrl"),
        )
        self.save()
        return {
            "bootstrap": bootstrap,
            "sourceSha": sha,
            "ownerCredentialFile": str(self.directory / "owner-credentials.json"),
        }

    def put_secret(self, arn, value):
        temporary = self.directory / "secret-payload.json"
        protected_json(temporary, value)
        try:
            self.aws(
                "secretsmanager",
                "put-secret-value",
                "--secret-id",
                arn,
                "--secret-string",
                "file://" + str(temporary),
            )
        finally:
            temporary.unlink(missing_ok=True)

    def apply(self):
        if self.installation.get("destroyed") or time.time() >= self.installation["budgetDeadline"]:
            raise ValueError(
                "Installation expired or was destroyed; do not extend the agreed billing window"
            )
        if (
            self.installation.get("stack")
            and self.installation.get("tracking")
            and self.installation.get("imagesBuildSha") == self.installation["sourceSha"]
        ):
            # A repeat apply must not rotate live leases, overwrite model lineage,
            # reset spending history or extend the original environment lifetime.
            self.build(self.installation["stack"]["cluster_project"], "cluster-apply")
            self.refresh()
            return {
                "url": self.installation["stack"]["url"],
                "budgetDeadline": self.installation["budgetDeadline"],
                "reapplied": True,
            }
        evidence = self.repo / "validation/v03-local.json"
        if not evidence.exists() or not json.loads(evidence.read_text()).get("allLocalGatesPassed"):
            raise ValueError(
                "Local acceptance gates must pass before opening the AWS billing window"
            )
        estimate = forecast(
            self.config.lifetime_hours,
            ceiling=self.config.aws_forecast_ceiling_usd,
            gpu_image_pulls=self.config.validation_gpu_image_pulls,
        )
        if estimate["estimate"] >= self.config.aws_forecast_ceiling_usd:
            raise ValueError("Full AWS forecast is not below $20")
        self.installation["forecast"] = estimate
        self.save()
        image_sha = self.reusable_image_build()
        if image_sha:
            print("Reusing verified immutable image build: " + image_sha, flush=True)
        else:
            image_sha = self.installation["sourceSha"]
            self.build(self.installation["bootstrap"]["build_project"])
        self.refresh()
        bucket = self.installation["bootstrap"]["buckets"]["reports"]
        images_file = self.directory / "images.json"
        self.command(
            [
                "aws",
                "s3",
                "cp",
                f"s3://{bucket}/builds/{image_sha}/images.json",
                str(images_file),
                "--only-show-errors",
            ]
        )
        images = json.loads(images_file.read_text())
        build_file = self.directory / "build.json"
        self.command(
            [
                "aws",
                "s3",
                "cp",
                f"s3://{bucket}/builds/{image_sha}/build.json",
                str(build_file),
                "--only-show-errors",
            ]
        )
        build = json.loads(build_file.read_text())
        if not build.get("linuxCpuKubernetesPassed") or build.get("sourceSha") != image_sha:
            raise ValueError("Linux Kubernetes acceptance did not pass")
        estimate = forecast(
            self.config.lifetime_hours,
            gpu_image_bytes=build["imageSizes"]["worker"],
            ceiling=self.config.aws_forecast_ceiling_usd,
            gpu_image_pulls=self.config.validation_gpu_image_pulls,
        )
        if estimate["estimate"] >= self.config.aws_forecast_ceiling_usd or estimate["hours"] < 2:
            raise ValueError("Remaining AWS allowance cannot cover acceptance and cleanup")
        self.installation["forecast"] = estimate
        self.installation["images"] = images
        self.installation["imagesBuildSha"] = image_sha
        self.installation.setdefault("billingStarted", time.time())
        self.installation["budgetDeadline"] = min(
            datetime.fromisoformat(self.installation["expiresAt"]).timestamp(),
            self.installation["budgetDeadline"],
            self.installation["billingStarted"] + estimate["hours"] * 3600,
        )
        self.save()
        self.build(self.installation["bootstrap"]["operations_project"], "foundation-apply")
        self.refresh()
        stack = self.installation["stack"]
        from cryptography.fernet import Fernet

        from qps.hosted.forge import setup_project

        tracking = setup_project(
            self.config.forge_entity,
            self.config.forge_project,
            read_key(self.config.forge_key_file),
            report_url=self.installation.get("tracking", {}).get("reportUrl"),
        )
        api_arn = self.installation["bootstrap"]["secrets"]["api"]
        versions = self.aws("secretsmanager", "list-secret-version-ids", "--secret-id", api_arn)[
            "Versions"
        ]
        existing = (
            json.loads(
                self.aws("secretsmanager", "get-secret-value", "--secret-id", api_arn)[
                    "SecretString"
                ]
            )
            if versions
            else {}
        )
        internal = existing.get("internal_token") or secrets.token_urlsafe(48)
        self.put_secret(
            self.installation["bootstrap"]["secrets"]["api"],
            {
                **existing,
                **(
                    {}
                    if existing.get("database_url")
                    else {
                        "database_password": existing.get("database_password")
                        or secrets.token_urlsafe(40)
                    }
                ),
                "internal_token": internal,
                "worker_signing_key": existing.get("worker_signing_key")
                or secrets.token_urlsafe(48),
                "session_key": existing.get("session_key") or Fernet.generate_key().decode(),
                "bucket": self.installation["bootstrap"]["buckets"]["artifacts"],
                "aws_region": self.config.region,
                "public_url": stack["url"],
                "environment_deadline": self.installation["budgetDeadline"],
                "cognito_issuer": stack["cognito_issuer"],
                "cognito_client_id": stack["cognito_client_id"],
                "cognito_domain": stack["cognito_domain"],
                "cpu_image": images["worker-cpu"],
                "gpu_image": images["worker"],
                "runpod_budget_usd": self.config.runpod_total_usd,
                "max_gpu_hourly_usd": self.config.runpod_hourly_usd,
                "forge_enabled": True,
                "forge_entity": self.config.forge_entity,
                "forge_project": self.config.forge_project,
                "forge_report_url": tracking["reportUrl"],
                "forge_report_public": tracking["public"],
                "wandb_api_key": read_key(self.config.forge_key_file),
            },
        )
        self.put_secret(
            self.installation["bootstrap"]["secrets"]["operator"],
            {"internal_token": internal, "runpod_api_key": read_key(self.config.runpod_key_file)},
        )
        self.installation["tracking"] = tracking
        self.save()
        self.build(stack["cluster_project"], "cluster-apply")
        self.wait_public()
        protected_json(
            self.directory / "owner-credentials.json",
            {
                **json.loads((self.directory / "owner-credentials.json").read_text()),
                "url": stack["url"],
                "expiresAt": datetime.fromtimestamp(
                    self.installation["budgetDeadline"], UTC
                ).isoformat(),
            },
        )
        return {
            "url": stack["url"],
            "expiresAt": datetime.fromtimestamp(
                self.installation["budgetDeadline"], UTC
            ).isoformat(),
            "credentialFile": str(self.directory / "owner-credentials.json"),
            "forecast": estimate,
        }

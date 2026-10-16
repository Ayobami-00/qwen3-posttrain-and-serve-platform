"""Scoped CodeBuild entry point. Uses its service role, never local/root credentials."""

import json
import os
import subprocess
import time
from pathlib import Path

import boto3
from botocore.config import Config as AWSConfig

from qps.hosted.install import protected_json


class Operations:
    def __init__(self):
        self.root = Path.cwd()
        self.region = os.environ["QPS_REGION"]
        self.s3 = boto3.client("s3", region_name=self.region)
        self.secrets = boto3.client("secretsmanager", region_name=self.region)
        self.bucket = os.environ["QPS_REPORTS_BUCKET"]
        self.configuration = json.loads(
            self.s3.get_object(Bucket=self.bucket, Key="installation/current.json")["Body"].read()
        )
        account = boto3.client("sts").get_caller_identity()["Account"]
        if account != self.configuration["accountId"] or account != os.environ["QPS_ACCOUNT_ID"]:
            raise ValueError("Installation account guard failed")
        if self.configuration["prefix"] != "qps-test":
            raise ValueError("Unexpected project ownership")
        self.log_dir = self.root / ".qps/operations"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save(self):
        self.s3.put_object(
            Bucket=self.bucket,
            Key="installation/current.json",
            Body=json.dumps(self.configuration).encode(),
        )

    def secret(self, name):
        return json.loads(
            self.secrets.get_secret_value(
                SecretId=self.configuration["bootstrap"]["secrets"][name]
            )["SecretString"]
        )

    def tf(self, phase, action):
        root = self.root / "infra/terraform/deployments/test" / phase
        (root / "backend.tf").write_text('terraform {\n backend "s3" {}\n}\n')
        config = self.configuration
        values = {"region": self.region}
        if phase == "foundation":
            values.update(
                account_id=config["accountId"],
                prefix=config["prefix"],
                buckets=config["bootstrap"]["buckets"],
                secrets=config["bootstrap"]["secrets"],
                provision_role_arn=config["bootstrap"]["provision_role_arn"],
                source_key=config["sourceKey"],
                source_sha=config["sourceSha"],
            )
        else:
            values["stack"] = config["stack"]
            if phase == "addons":
                values["logs_bucket"] = config["bootstrap"]["buckets"]["logs"]
            else:
                values.update(images=config["images"], secrets=config["bootstrap"]["secrets"])
        protected_json(root / "run.tfvars.json", values)
        init = [
            "terraform",
            "init",
            "-input=false",
            f"-backend-config=bucket={config['bootstrap']['buckets']['state']}",
            f"-backend-config=key=test/{phase}.tfstate",
            f"-backend-config=region={self.region}",
            "-backend-config=use_lockfile=true",
            "-backend-config=encrypt=true",
        ]
        subprocess.run(init, cwd=root, check=True)
        args = ["terraform", action, "-input=false", "-var-file=run.tfvars.json"]
        if action != "plan":
            args.append("-auto-approve")
        # Terraform receives no secret values. Its logs and state contain infrastructure identities.
        subprocess.run(args, cwd=root, check=True)
        if phase == "foundation" and action == "apply":
            output = subprocess.check_output(
                ["terraform", "output", "-json", "stack"], cwd=root, text=True
            )
            config["stack"] = json.loads(output)
            self.save()

    def database(self):
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import URL, make_url

        from qps.hosted.database import Repository
        from qps.hosted.settings import Settings

        config = self.configuration
        api_arn = config["bootstrap"]["secrets"]["api"]
        api = self.secret("api")
        password = api.get("database_password") or make_url(api["database_url"]).password
        master = json.loads(
            self.secrets.get_secret_value(SecretId=config["stack"]["database_master_secret"])[
                "SecretString"
            ]
        )
        address = config["stack"]["database_host"]
        admin_url = URL.create(
            "postgresql+psycopg",
            username=master["username"],
            password=master["password"],
            host=address,
            database="qps",
            query={"sslmode": "require"},
        )
        engine = create_engine(admin_url)
        # Password is sent only as a bound DB value; identifiers are fixed application constants.
        with engine.begin() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname='qpsapp'")
            ).scalar()
            if not exists:
                connection.execute(text("CREATE ROLE qpsapp LOGIN"))
            connection.execute(
                text("SELECT set_config('qps.install_password', :password, true)"),
                {"password": password},
            )
            connection.execute(
                text(
                    "DO $$ BEGIN EXECUTE format('ALTER ROLE qpsapp PASSWORD %L', current_setting('qps.install_password')); END $$"
                )
            )
        # Migrations use the master role, then app privileges are restricted to the qps schema.
        from alembic import command
        from alembic.config import Config

        alembic = Config(str(self.root / "python/alembic.ini"))
        with engine.begin() as connection:
            alembic.attributes["connection"] = connection
            command.upgrade(alembic, "head")
            connection.execute(text("GRANT USAGE ON SCHEMA public TO qpsapp"))
            connection.execute(
                text(
                    "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO qpsapp"
                )
            )
            connection.execute(
                text("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO qpsapp")
            )
        api.pop("database_password", None)
        app_url = URL.create(
            "postgresql+psycopg",
            username="qpsapp",
            password=password,
            host=address,
            database="qps",
            query={"sslmode": "require"},
        )
        api["database_url"] = app_url.render_as_string(hide_password=False)
        self.secrets.put_secret_value(SecretId=api_arn, SecretString=json.dumps(api))
        settings = Settings.model_validate(
            {k: v for k, v in api.items() if k in Settings.model_fields}
        )
        Repository(settings).budget()  # Initializes the fixed, transactionally locked allowance.
        owner = self.secret("owner")
        boto3.client("cognito-idp", region_name=self.region).admin_set_user_password(
            UserPoolId=config["stack"]["cognito_pool_id"],
            Username=owner["username"],
            Password=owner["password"],
            Permanent=True,
        )

    def backup(self, restore=False):
        from qps.hosted.database import Repository
        from qps.hosted.settings import Settings

        secret = self.secret("api")
        repo = Repository(
            Settings.model_validate({k: v for k, v in secret.items() if k in Settings.model_fields})
        )
        key = "backups/metadata-latest.json"
        if restore:
            repo.restore_metadata(
                json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())
            )
            from sqlalchemy import update

            from qps.hosted.database import budgets

            with repo.transaction() as connection:
                connection.execute(
                    update(budgets)
                    .where(budgets.c.id == "project")
                    .values(slots=repo.settings.max_gpu_allocations)
                )
        else:
            # One snapshot across all tables, even while another API replica
            # commits submissions or metrics during the backup.
            if repo.engine.dialect.name == "postgresql":
                repo.engine = repo.engine.execution_options(isolation_level="REPEATABLE READ")
            data = repo.export_metadata()
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(data).encode(),
                ServerSideEncryption="AES256",
            )
            print("Metadata backup saved and read verified", flush=True)
            assert (
                json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()) == data
            )

    def drain(self):
        from sqlalchemy import update

        from qps.contracts import TERMINAL
        from qps.hosted.database import Repository, budgets, deployments, runs
        from qps.hosted.settings import Settings

        values = self.secret("api")
        repo = Repository(
            Settings.model_validate({k: v for k, v in values.items() if k in Settings.model_fields})
        )
        with repo.transaction() as connection:
            connection.execute(update(budgets).where(budgets.c.id == "project").values(slots=0))
        for run in repo.list(runs):
            if run["status"] not in TERMINAL:
                repo.action(run["id"], "cancel")
        for deployment in repo.list(deployments):
            if deployment["status"] not in TERMINAL:
                repo.deployment_action(deployment["id"], "stop")

    def capture_volumes(self):
        import base64

        import httpx

        from qps.hosted.cleanup import monitoring_inventory

        certificate = self.log_dir / "cluster-ca.crt"
        certificate.write_bytes(base64.b64decode(self.configuration["stack"]["cluster_ca"]))
        result = subprocess.check_output(
            [
                "aws",
                "eks",
                "get-token",
                "--cluster-name",
                self.configuration["stack"]["cluster_name"],
                "--region",
                self.region,
            ]
        )
        token = json.loads(result)["status"]["token"]
        with httpx.Client(
            base_url=self.configuration["stack"]["cluster_endpoint"],
            verify=str(certificate),
            headers={"Authorization": "Bearer " + token},
            timeout=20,
        ) as client:
            inventory = monitoring_inventory(client)
        self.s3.put_object(
            Bucket=self.bucket,
            Key="cleanup/monitoring-volumes.json",
            Body=json.dumps(inventory).encode(),
        )

    def cleanup_volumes(self):
        from qps.hosted.cleanup import delete_detached_volumes

        try:
            inventory = json.loads(
                self.s3.get_object(Bucket=self.bucket, Key="cleanup/monitoring-volumes.json")[
                    "Body"
                ].read()
            )
        except self.s3.exceptions.NoSuchKey:
            return
        delete_detached_volumes(boto3.client("ec2", region_name=self.region), inventory)
        self.s3.put_object(
            Bucket=self.bucket,
            Key="cleanup/monitoring-volumes-result.json",
            Body=json.dumps({"deletedOrAbsent": inventory}).encode(),
        )

    def janitor(self):
        response = boto3.client(
            "lambda", config=AWSConfig(read_timeout=150, retries={"max_attempts": 0})
        ).invoke(
            FunctionName=self.configuration["bootstrap"]["janitor_arn"],
            Payload=json.dumps({"force": True, "cleanupOnly": True}).encode(),
        )
        result = json.loads(response["Payload"].read())
        self.configuration["cleanupPending"] = bool(response.get("FunctionError")) or any(
            not a.get("absent") for a in result.get("allocations", [])
        )

    def cluster_job(self, operation):
        client = boto3.client("codebuild", region_name=self.region)
        build = client.start_build(
            projectName=self.configuration["stack"]["cluster_project"],
            environmentVariablesOverride=[
                {"name": "QPS_OPERATION", "value": operation, "type": "PLAINTEXT"}
            ],
        )["build"]
        deadline = time.time() + 2400
        while time.time() < deadline:
            status = client.batch_get_builds(ids=[build["id"]])["builds"][0]["buildStatus"]
            if status != "IN_PROGRESS":
                if status != "SUCCEEDED":
                    raise RuntimeError("Private cluster operations job failed: " + build["id"])
                return
            time.sleep(15)
        raise TimeoutError("Private cluster job exceeded bounded wait")

    def execute(self, action):
        if (
            action == "destroy"
            and self.configuration.get("destroyed")
            and not self.configuration.get("cleanupPending")
        ):
            if not self.configuration.get("credentialsRemoved"):
                self.close_credentials()
            return
        if action in {"plan", "foundation-apply"}:
            self.tf("foundation", "plan" if action == "plan" else "apply")
        elif action == "cluster-apply":
            self.database()
            self.tf("addons", "apply")
            self.tf("platform", "apply")
            self.capture_volumes()
        elif action == "cluster-destroy":
            self.drain()
            self.backup()
            self.capture_volumes()
            self.tf("platform", "destroy")
            self.tf("addons", "destroy")
        elif action in {"backup", "restore"}:
            self.backup(restore=action == "restore")
        elif action == "destroy":
            self.janitor()
            # The independent janitor and allocation ledger remain after EKS removal.
            # A provider outage must not keep the CPU billing window open indefinitely.
            self.save()
            if self.configuration.get("stack"):
                try:
                    self.cluster_job("cluster-destroy")
                except Exception:
                    # Removing EKS remains necessary after a failed Helm uninstall; all provider
                    # allocations were already independently confirmed absent above.
                    print(
                        "Cluster uninstall incomplete; deleting owned infrastructure to bound cost",
                        flush=True,
                    )
            if self.configuration.get("stack") or self.configuration.get("billingStarted"):
                # Terraform may have created billable resources before foundation outputs
                # were saved. Its remote state still owns those partial resources.
                self.tf("foundation", "destroy")
            self.cleanup_volumes()
            self.janitor()
            self.configuration["destroyed"] = True
            self.configuration["destroyedAt"] = time.time()
            self.save()
            if not self.configuration["cleanupPending"]:
                self.close_credentials()
        elif action == "cleanup-secrets":
            self.close_credentials()
        elif action == "status":
            print(
                json.dumps(
                    {
                        "destroyed": self.configuration.get("destroyed"),
                        "expiresAt": self.configuration["expiresAt"],
                    }
                )
            )
        else:
            raise ValueError("Unsupported managed operation")

    def close_credentials(self):
        if not self.configuration.get("destroyed") or self.configuration.get("cleanupPending"):
            raise RuntimeError("Credential cleanup requires confirmed compute teardown")
        for arn in self.configuration["bootstrap"]["secrets"].values():
            try:
                record = self.secrets.describe_secret(SecretId=arn)
            except self.secrets.exceptions.ResourceNotFoundException:
                continue
            tags = {tag["Key"]: tag["Value"] for tag in record.get("Tags", [])}
            if (
                not record["Name"].startswith("qps-test/")
                or tags.get("Project") != "qps"
                or tags.get("Environment") != "test"
            ):
                raise RuntimeError("Refusing to delete a secret without verified project ownership")
            self.secrets.delete_secret(SecretId=arn, ForceDeleteWithoutRecovery=True)
        boto3.client("events").disable_rule(Name="qps-test-janitor")
        self.configuration["credentialsRemoved"] = True
        self.save()


if __name__ == "__main__":
    try:
        Operations().execute(os.environ.get("QPS_OPERATION", "status"))
    except Exception as exc:
        # Never print SQLAlchemy exception strings: they may include bound secret values.
        print("Managed operation failed: " + type(exc).__name__, flush=True)
        raise SystemExit(1) from None

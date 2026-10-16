"""S3 artifact commitment: verify staging bytes, copy immutable versions, publish last."""

import base64
import hashlib
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from sqlalchemy import insert, select, update

from qps.artifacts import validate_manifest, verify_directory
from qps.contracts import ARTIFACT_KINDS
from qps.hosted.database import (
    Conflict,
    allocations,
    artifacts,
    cost_ledger,
    provenance,
    tombstones,
    uploads,
)
from qps.util import atomic_json, canonical, digest


class ObjectStore:
    def __init__(self, settings, repository, client=None):
        self.settings = settings
        self.repo = repository
        self.client = client or boto3.client(
            "s3",
            region_name=settings.aws_region,
            endpoint_url=settings.s3_endpoint,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=5,
                read_timeout=30,
                retries={"total_max_attempts": 3, "mode": "standard"},
            ),
        )
        self.bucket = settings.bucket
        self.prefix = settings.object_prefix.strip("/")

    def key(self, *parts):
        return "/".join([self.prefix, *parts])

    def _put_url(self, key, info, expires):
        checksum = base64.b64encode(bytes.fromhex(info["sha256"])).decode()
        parameters = {
            "Bucket": self.bucket,
            "Key": key,
            "ChecksumSHA256": checksum,
            "ContentLength": info["size"],
        }
        return {
            "url": self.client.generate_presigned_url(
                "put_object", Params=parameters, ExpiresIn=expires
            ),
            "headers": {"x-amz-checksum-sha256": checksum, "Content-Length": str(info["size"])},
        }

    def start(self, allocation, manifest):
        validate_manifest(manifest)
        if manifest["kind"] not in ARTIFACT_KINDS:
            raise ValueError("Unsupported artifact kind")
        if allocation["kind"] != "training" or manifest["run_id"] != allocation["owner_id"]:
            raise Conflict("Artifact does not belong to this training allocation")
        size = sum(info["size"] for info in manifest["files"].values())
        if size > self.settings.upload_limit_bytes or any(
            v["size"] > 5 * 1024**3 for v in manifest["files"].values()
        ):
            raise ValueError("Artifact exceeds supported upload size (5 GiB/file)")
        sha = hashlib.sha256((canonical(manifest) + "\n").encode()).hexdigest()
        with self.repo.transaction() as connection:
            self.repo.assert_live(connection, allocation)
            old = (
                connection.execute(select(artifacts).where(artifacts.c.id == manifest["id"]))
                .mappings()
                .first()
            )
            if old:
                if old["manifest_sha"] != sha or old["run_id"] != manifest["run_id"]:
                    raise Conflict("Artifact ID is already committed with different contents")
                return {"committed": True, "artifactId": manifest["id"], "manifestSha": sha}
            old_upload = (
                connection.execute(
                    select(uploads).where(
                        (uploads.c.allocation_id == allocation["id"])
                        & (uploads.c.manifest_sha == sha)
                        & (uploads.c.state == "staging")
                        & (uploads.c.expires > time.time())
                    )
                )
                .mappings()
                .first()
            )
            upload_id = old_upload["id"] if old_upload else "upload-" + uuid.uuid4().hex
            if not old_upload:
                connection.execute(
                    insert(uploads).values(
                        id=upload_id,
                        allocation_id=allocation["id"],
                        manifest=manifest,
                        manifest_sha=sha,
                        state="staging",
                        created=time.time(),
                        expires=min(
                            allocation["deadline"], time.time() + self.settings.upload_url_seconds
                        ),
                    )
                )
        expires = max(
            1,
            min(
                self.settings.upload_url_seconds,
                int(
                    (old_upload["expires"] if old_upload else allocation["deadline"]) - time.time()
                ),
            ),
        )
        return {
            "committed": False,
            "uploadId": upload_id,
            "manifestSha": sha,
            "files": {
                name: self._put_url(
                    self.key("staging", allocation["id"], upload_id, name), info, expires
                )
                for name, info in manifest["files"].items()
            },
        }

    def _publication_lease(self, allocation, upload_id):
        with self.repo.transaction() as c:
            self.repo.assert_live(c, allocation)
            record = self.repo.one(uploads, upload_id, c)
            c.execute(
                update(uploads)
                .where(uploads.c.id == upload_id)
                .values(
                    expires=max(
                        record["expires"], time.time() + self.settings.publication_grace_seconds
                    )
                )
            )

    def commit(self, allocation, upload_id):
        upload = self.repo.one(uploads, upload_id)
        if upload["allocation_id"] != allocation["id"] or upload["expires"] < time.time():
            raise Conflict("Upload expired or belongs to another allocation")
        manifest = upload["manifest"]
        if upload["state"] == "committed":
            return self.repo.one(artifacts, manifest["id"])
        objects = {}
        retention = "reports" if manifest["kind"].endswith("report") else "weights"
        for name, info in manifest["files"].items():
            self._publication_lease(allocation, upload_id)
            staging = self.key("staging", allocation["id"], upload_id, name)
            response = self.client.get_object(Bucket=self.bucket, Key=staging)
            checksum = hashlib.sha256()
            size = 0
            started = time.monotonic()
            try:
                for chunk in iter(lambda: response["Body"].read(1024 * 1024), b""):
                    if time.monotonic() - started > 90:
                        raise TimeoutError("Artifact verification exceeded its bounded read window")
                    checksum.update(chunk)
                    size += len(chunk)
            finally:
                response["Body"].close()
            if size != info["size"] or checksum.hexdigest() != info["sha256"]:
                raise ValueError(f"Staged artifact checksum mismatch: {name}")
            final = self.key("artifacts", manifest["id"], "files", name)
            self._publication_lease(allocation, upload_id)
            copied = self.client.copy_object(
                Bucket=self.bucket,
                Key=final,
                CopySource={"Bucket": self.bucket, "Key": staging},
                MetadataDirective="REPLACE",
                Metadata={"sha256": info["sha256"]},
                ServerSideEncryption="AES256",
                TaggingDirective="REPLACE",
                Tagging="qps-retention=" + retention,
            )
            objects[name] = {"key": final, "versionId": copied.get("VersionId")}
        manifest_key = self.key("artifacts", manifest["id"], "manifest.json")
        body = (canonical(manifest) + "\n").encode()
        self._publication_lease(allocation, upload_id)
        response = self.client.put_object(
            Bucket=self.bucket,
            Key=manifest_key,
            Body=body,
            ServerSideEncryption="AES256",
            ContentType="application/json",
            Tagging="qps-retention=" + retention,
        )
        objects["@manifest"] = {"key": manifest_key, "versionId": response.get("VersionId")}
        # Publication may take time. Recheck fencing *inside* the registration transaction.
        with self.repo.transaction() as connection:
            self.repo.assert_live(connection, allocation)
            current = self.repo.one(allocations, allocation["id"], connection)
            if (
                current["state"] == "terminated"
                or current["deadline"] < time.time()
                or current["desired"] != "run"
            ):
                raise Conflict("Worker was fenced before artifact commitment")
            from qps.hosted.database import runs

            owner = self.repo.one(runs, allocation["owner_id"], connection)
            if owner["attempt"] != allocation["generation"]:
                raise Conflict("Worker attempt was superseded")
            old = (
                connection.execute(select(artifacts).where(artifacts.c.id == manifest["id"]))
                .mappings()
                .first()
            )
            if old:
                if old["manifest_sha"] != upload["manifest_sha"]:
                    raise Conflict("Artifact identity conflict")
                return dict(old)
            total = sum(connection.execute(select(artifacts.c.size)).scalars())
            size = sum(v["size"] for v in manifest["files"].values())
            if total + size > self.settings.upload_limit_bytes:
                raise Conflict("Project artifact storage allowance exhausted")
            connection.execute(
                insert(artifacts).values(
                    id=manifest["id"],
                    run_id=manifest["run_id"],
                    attempt=allocation["attempt"],
                    kind=manifest["kind"],
                    manifest_sha=upload["manifest_sha"],
                    manifest=manifest,
                    objects=objects,
                    size=size,
                    created=time.time(),
                )
            )
            owner = self.repo.one(runs, allocation["owner_id"], connection)
            connection.execute(
                insert(provenance).values(
                    id=manifest["id"],
                    recipe=owner["resolved"].get("run", owner["submission"]["run"]),
                )
            )
            self.repo.add_reference(
                connection, allocation["owner_id"], manifest["id"], "experiment"
            )
            connection.execute(
                update(uploads).where(uploads.c.id == upload_id).values(state="committed")
            )
        for name in manifest["files"]:
            self._publication_lease(allocation, upload_id)
            self.client.delete_object(
                Bucket=self.bucket, Key=self.key("staging", allocation["id"], upload_id, name)
            )
        return self.repo.one(artifacts, manifest["id"])

    def download_descriptor(self, artifact_id):
        with self.repo.connection() as connection:
            if connection.execute(select(tombstones).where(tombstones.c.id == artifact_id)).first():
                raise Conflict("Artifact is deleting")
        record = self.repo.one(artifacts, artifact_id)
        retention_days = 30 if record["kind"].endswith("report") else 7
        if record["created"] + retention_days * 86400 <= time.time():
            raise Conflict("Artifact retention expired; prepare a new run")
        urls = {}
        for name, value in record["objects"].items():
            parameters = {"Bucket": self.bucket, "Key": value["key"]}
            if value.get("versionId"):
                parameters["VersionId"] = value["versionId"]
            urls[name] = self.client.generate_presigned_url(
                "get_object", Params=parameters, ExpiresIn=900
            )
        return {
            "id": artifact_id,
            "manifest": record["manifest"],
            "manifestSha": record["manifest_sha"],
            "urls": urls,
        }

    def read_json(self, artifact_id, name="report.json"):
        record = self.repo.one(artifacts, artifact_id)
        info = record["manifest"]["files"].get(name)
        if not info or info["size"] > 32 * 1024**2:
            raise ValueError("Report file absent or too large")
        value = record["objects"][name]
        kwargs = {"Bucket": self.bucket, "Key": value["key"]}
        if value.get("versionId"):
            kwargs["VersionId"] = value["versionId"]
        response = self.client.get_object(**kwargs)
        try:
            body = response["Body"].read()
        finally:
            response["Body"].close()
        if hashlib.sha256(body).hexdigest() != info["sha256"]:
            raise ValueError("Report checksum mismatch")
        return json.loads(body)

    def _recovery_record(self, directory, identity, value):
        from botocore.exceptions import ClientError

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self.key(directory, identity + ".json"),
                Body=(canonical(value) + "\n").encode(),
                ServerSideEncryption="AES256",
                Tagging="qps-retention=reports",
                IfNoneMatch="*",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
                raise

    def deletion_tombstones(self, plan):
        for identity in [*plan["owners"], *plan["artifacts"]]:
            value = {
                "id": identity,
                "kind": "artifact" if identity.startswith("art-") else plan["kind"],
                "created": time.time(),
            }
            self._recovery_record("tombstones", identity, value)

    def recovery_ledger(self):
        result = {"tombstones": [], "cost_ledger": []}
        for category in result:
            directory = "accounting" if category == "cost_ledger" else category
            for page in self.client.get_paginator("list_objects_v2").paginate(
                Bucket=self.bucket, Prefix=self.key(directory) + "/"
            ):
                for item in page.get("Contents", []):
                    if item["Size"] > 4096:
                        raise ValueError("Recovery ledger record exceeds its limit")
                    response = self.client.get_object(Bucket=self.bucket, Key=item["Key"])
                    try:
                        result[category].append(json.loads(response["Body"].read()))
                    finally:
                        response["Body"].close()
        return result

    def purge_prefix(self, prefix, *, exact=False):
        # Caller constructs prefixes from trusted operation identities. Never accept a raw UI key.
        if not prefix.startswith(self.prefix + "/") or prefix == self.prefix + "/":
            raise ValueError("Refusing unscoped object deletion")
        for page in self.client.get_paginator("list_multipart_uploads").paginate(
            Bucket=self.bucket, Prefix=prefix
        ):
            for upload in page.get("Uploads", []):
                if not exact or upload["Key"] == prefix:
                    self.client.abort_multipart_upload(
                        Bucket=self.bucket, Key=upload["Key"], UploadId=upload["UploadId"]
                    )
        # Deleting while paginating versions can skip continuation keys. Repeat first-page sweeps.
        while True:
            page = self.client.list_object_versions(Bucket=self.bucket, Prefix=prefix, MaxKeys=1000)
            versions = [
                {"Key": v["Key"], "VersionId": v["VersionId"]}
                for v in page.get("Versions", []) + page.get("DeleteMarkers", [])
                if not exact or v["Key"] == prefix
            ]
            if not versions:
                break
            result = self.client.delete_objects(
                Bucket=self.bucket, Delete={"Objects": versions, "Quiet": True}
            )
            if result.get("Errors"):
                raise RuntimeError("S3 version deletion incomplete")
        if self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1).get(
            "KeyCount", 0
        ):
            raise RuntimeError("Object sweep is incomplete")

    def ledger(self, allocation):
        if allocation["state"] == "terminated":
            value = self.repo.one(cost_ledger, allocation["id"])
            self._recovery_record("accounting", allocation["id"], value)
        value = {
            key: allocation[key]
            for key in (
                "id",
                "pod_id",
                "created",
                "deadline",
                "lease_until",
                "state",
                "owner_id",
                "kind",
                "profile",
                "gpu_count",
                "worker_group",
            )
        }
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.key("allocations", allocation["id"] + ".json"),
            Body=(canonical(value) + "\n").encode(),
            ServerSideEncryption="AES256",
            ContentType="application/json",
        )


def download_artifact(descriptor, destination: Path, client):
    """Client-side verified cache; manifest is checked before any file path is joined."""
    manifest = descriptor["manifest"]
    validate_manifest(manifest)
    if (
        hashlib.sha256((canonical(manifest) + "\n").encode()).hexdigest()
        != descriptor["manifestSha"]
    ):
        raise ValueError("Download manifest identity mismatch")
    if destination.exists():
        try:
            if digest(destination / "manifest.json") == descriptor["manifestSha"]:
                verify_directory(destination / "files", manifest)
                return destination
        except (OSError, ValueError):
            pass
        raise ValueError("Existing artifact cache is corrupt; refusing to overwrite it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".download-", dir=destination.parent))
    try:
        for name, info in manifest["files"].items():
            target = temporary / "files" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256()
            size = 0
            with client.stream("GET", descriptor["urls"][name], headers={}) as response:
                response.raise_for_status()
                with target.open("wb") as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > info["size"]:
                            raise ValueError("Download exceeds declared size")
                        checksum.update(chunk)
                        stream.write(chunk)
            if size != info["size"] or checksum.hexdigest() != info["sha256"]:
                raise ValueError(f"Download checksum mismatch: {name}")
        verify_directory(temporary / "files", manifest)
        atomic_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination

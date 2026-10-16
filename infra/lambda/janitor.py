"""Independent, fail-closed cleanup. Never discovers ownership from a name prefix alone."""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

import boto3


def request(method, path, token):
    req = urllib.request.Request(
        "https://api.runpod.io/v2" + path,
        method=method,
        headers={"Authorization": "Bearer " + token},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            return json.load(response) if method == "GET" else None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError("Runpod cleanup request failed") from None


def expired(record, now):
    return (
        record["deadline"] <= now
        or record["lease_until"] + 60 <= now
        or record["state"] in {"terminating", "terminated"}
    )


def cleanup(records, pods, token, now, force=False):
    outcomes = []
    by_name = {pod["name"]: pod for pod in pods}
    for record in records:
        if record.get("profile") == "cpu-fixture" or not (
            force or expired(record, now)
        ):
            continue
        name = os.environ["PREFIX"] + "-" + record["id"]
        pod = by_name.get(name)
        if not pod:
            if (
                record.get("pod_id")
                and request("GET", "/pods/" + record["pod_id"], token) is not None
            ):
                outcomes.append(
                    {"allocationId": record["id"], "identityConflict": True}
                )
            elif (
                not record.get("pod_id")
                and record["state"] in {"reserved", "creating"}
                and now < record["deadline"] + 60
            ):
                outcomes.append(
                    {
                        "allocationId": record["id"],
                        "absent": False,
                        "uncertainCreate": True,
                    }
                )
            else:
                outcomes.append({"allocationId": record["id"], "absent": True})
            continue
        # Both the durable allocation intent and the provider's exact identity must agree.
        if record.get("pod_id") and pod["id"] != record["pod_id"]:
            outcomes.append({"allocationId": record["id"], "identityConflict": True})
            continue
        request("DELETE", "/pods/" + pod["id"], token)
        absent = request("GET", "/pods/" + pod["id"], token) is None
        outcomes.append(
            {"allocationId": record["id"], "podId": pod["id"], "absent": absent}
        )
    return outcomes


def handler(event, context):
    now = time.time()
    s3 = boto3.client("s3")
    report_bucket = os.environ["REPORT_BUCKET"]
    try:
        installation = json.loads(
            s3.get_object(Bucket=report_bucket, Key="installation/current.json")[
                "Body"
            ].read()
        )
    except s3.exceptions.NoSuchKey:
        # Bootstrap may exist before any billable environment or Runpod allocation.
        return {"state": "not-installed"}
    if installation.get("destroyed") and not installation.get("cleanupPending"):
        return {"state": "complete"}
    secret = json.loads(
        boto3.client("secretsmanager").get_secret_value(
            SecretId=os.environ["SECRET_ARN"]
        )["SecretString"]
    )
    deadline = min(
        datetime.fromisoformat(os.environ["EXPIRY"]).timestamp(),
        installation["budgetDeadline"],
    )
    force = now >= deadline or bool(event.get("force"))
    records = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=os.environ["ARTIFACT_BUCKET"],
        Prefix=os.environ["ENVIRONMENT"] + "/allocations/",
    ):
        for item in page.get("Contents", []):
            records.append(
                json.loads(
                    s3.get_object(
                        Bucket=os.environ["ARTIFACT_BUCKET"], Key=item["Key"]
                    )["Body"].read()
                )
            )
    pods = []
    cursor = ""
    for _ in range(100):
        payload = request("GET", "/pods?limit=100" + cursor, secret["runpod_api_key"])
        pods.extend(payload.get("pods", []))
        next_cursor = payload.get("pagination", {}).get("nextCursor") or payload.get(
            "nextCursor"
        )
        if not next_cursor:
            break
        cursor = "&cursor=" + urllib.parse.quote(next_cursor, safe="")
    outcomes = cleanup(records, pods, secret["runpod_api_key"], now, force)
    retired = {
        os.environ["PREFIX"] + "-" + x["allocationId"]
        for x in outcomes
        if x.get("absent")
    }
    if retired:
        registries = request("GET", "/registries", secret["runpod_api_key"]).get(
            "registries", []
        )
        for registry in registries:
            if registry["name"] in retired:
                request(
                    "DELETE", "/registries/" + registry["id"], secret["runpod_api_key"]
                )
    report = {"checked": now, "expired": force, "allocations": outcomes}
    expected = {x["id"] for x in records if x.get("profile") != "cpu-fixture"}
    if (
        installation.get("destroyed")
        and installation.get("cleanupPending")
        and {x["allocationId"] for x in outcomes} == expected
        and all(x.get("absent") for x in outcomes)
    ):
        installation["cleanupPending"] = False
        s3.put_object(
            Bucket=report_bucket,
            Key="installation/current.json",
            Body=json.dumps(installation).encode(),
        )
        boto3.client("codebuild").start_build(
            projectName=os.environ["OPERATIONS_PROJECT"],
            environmentVariablesOverride=[
                {
                    "name": "QPS_OPERATION",
                    "value": "cleanup-secrets",
                    "type": "PLAINTEXT",
                }
            ],
        )
    if force and not event.get("cleanupOnly") and not installation.get("destroyed"):
        try:
            prior = json.loads(
                s3.get_object(Bucket=report_bucket, Key="cleanup/teardown.json")[
                    "Body"
                ].read()
            )
        except s3.exceptions.NoSuchKey:
            prior = {}
        active = False
        if prior.get("buildId"):
            builds = boto3.client("codebuild").batch_get_builds(ids=[prior["buildId"]])[
                "builds"
            ]
            active = bool(builds and builds[0]["buildStatus"] == "IN_PROGRESS")
        if not active and (not prior or now - prior.get("started", 0) > 300):
            build = boto3.client("codebuild").start_build(
                projectName=os.environ["OPERATIONS_PROJECT"],
                environmentVariablesOverride=[
                    {"name": "QPS_OPERATION", "value": "destroy", "type": "PLAINTEXT"}
                ],
                idempotencyToken=str(int(now // 300)),
            )["build"]
            s3.put_object(
                Bucket=report_bucket,
                Key="cleanup/teardown.json",
                Body=json.dumps({"buildId": build["id"], "started": now}).encode(),
            )
    s3.put_object(
        Bucket=report_bucket,
        Key="cleanup/latest.json",
        Body=json.dumps(report).encode(),
    )
    return report

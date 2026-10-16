"""Only delete volumes captured from verified, owned monitoring resources."""

from botocore.exceptions import ClientError


def monitoring_inventory(client):
    def get(path):
        response = client.get(path)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return response.json()["items"]

    owned = {
        x["metadata"]["uid"]
        for x in get("/apis/monitoring.coreos.com/v1/namespaces/monitoring/prometheuses")
        if x["metadata"].get("labels", {}).get("release") == "qps-monitoring"
    }
    for item in get("/apis/apps/v1/namespaces/monitoring/statefulsets"):
        meta = item["metadata"]
        if meta.get("annotations", {}).get("meta.helm.sh/release-name") == "qps-loki" or any(
            o["uid"] in owned for o in meta.get("ownerReferences", [])
        ):
            owned.add(meta["uid"])
    volumes = []
    for claim in get("/api/v1/namespaces/monitoring/persistentvolumeclaims"):
        meta = claim["metadata"]
        if meta.get("annotations", {}).get("meta.helm.sh/release-name") not in {
            "qps-monitoring",
            "qps-loki",
        } and not any(o["uid"] in owned for o in meta.get("ownerReferences", [])):
            continue
        name = claim["spec"].get("volumeName")
        if name:
            response = client.get("/api/v1/persistentvolumes/" + name)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            handle = response.json()["spec"].get("csi", {}).get("volumeHandle")
            if handle and handle.startswith("vol-"):
                volumes.append(
                    {"volumeId": handle, "claimUid": meta["uid"], "claimName": meta["name"]}
                )
    return volumes


def delete_detached_volumes(ec2, inventory):
    for item in inventory:
        try:
            volume = ec2.describe_volumes(VolumeIds=[item["volumeId"]])["Volumes"][0]
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "InvalidVolume.NotFound":
                continue
            raise
        tags = {t["Key"]: t["Value"] for t in volume.get("Tags", [])}
        if (
            tags.get("Project") != "qps"
            or tags.get("Environment") != "test"
            or volume["Attachments"]
            or volume["State"] != "available"
        ):
            raise RuntimeError("Volume cleanup requires verified project ownership and detachment")
        ec2.delete_volume(VolumeId=item["volumeId"])

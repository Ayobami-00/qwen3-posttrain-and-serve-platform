import boto3
import httpx
import pytest
from moto import mock_aws

from qps.hosted.cleanup import delete_detached_volumes, monitoring_inventory


def test_volume_inventory_follows_owned_resource_uids_and_ignores_foreign_claims():
    def handle(request):
        path = request.url.path
        if path.endswith("/prometheuses"):
            data = {
                "items": [
                    {"metadata": {"uid": "prom-owned", "labels": {"release": "qps-monitoring"}}}
                ]
            }
        elif path.endswith("/statefulsets"):
            data = {
                "items": [
                    {"metadata": {"uid": "sts-owned", "ownerReferences": [{"uid": "prom-owned"}]}}
                ]
            }
        elif path.endswith("/persistentvolumeclaims"):
            data = {
                "items": [
                    {
                        "metadata": {
                            "uid": "claim-ours",
                            "name": "prom",
                            "ownerReferences": [{"uid": "sts-owned"}],
                        },
                        "spec": {"volumeName": "pv-ours"},
                    },
                    {
                        "metadata": {
                            "uid": "claim-foreign",
                            "name": "qps-monitoring-other",
                            "ownerReferences": [{"uid": "sts-foreign"}],
                        },
                        "spec": {"volumeName": "pv-foreign"},
                    },
                ]
            }
        else:
            assert path == "/api/v1/persistentvolumes/pv-ours"
            data = {"spec": {"csi": {"volumeHandle": "vol-owned"}}}
        return httpx.Response(200, json=data)

    with httpx.Client(
        base_url="https://kubernetes.invalid", transport=httpx.MockTransport(handle)
    ) as client:
        assert monitoring_inventory(client) == [
            {"volumeId": "vol-owned", "claimUid": "claim-ours", "claimName": "prom"}
        ]


@mock_aws
def test_volume_deletion_requires_project_tags_and_is_idempotent():
    ec2 = boto3.client(
        "ec2", region_name="eu-west-2", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    own = ec2.create_volume(
        AvailabilityZone="eu-west-2a",
        Size=1,
        TagSpecifications=[
            {
                "ResourceType": "volume",
                "Tags": [
                    {"Key": "Project", "Value": "qps"},
                    {"Key": "Environment", "Value": "test"},
                ],
            }
        ],
    )["VolumeId"]
    other = ec2.create_volume(AvailabilityZone="eu-west-2a", Size=1)["VolumeId"]
    inventory = [{"volumeId": own, "claimUid": "ours"}]
    delete_detached_volumes(ec2, inventory)
    delete_detached_volumes(ec2, inventory)
    with pytest.raises(RuntimeError, match="ownership"):
        delete_detached_volumes(ec2, [{"volumeId": other}])
    assert ec2.describe_volumes(VolumeIds=[other])["Volumes"][0]["VolumeId"] == other

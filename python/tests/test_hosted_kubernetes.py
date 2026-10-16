import json

import httpx
import pytest

from qps.hosted.kubernetes import Kubernetes


def test_owned_resource_deletion_waits_for_uid_scoped_finalizers():
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "metadata": {
                        "uid": "original-uid",
                        "labels": {"app.kubernetes.io/part-of": "qps"},
                    }
                },
            )
        assert json.loads(request.content)["preconditions"] == {"uid": "original-uid"}
        return httpx.Response(202, json={})

    transport = object.__new__(Kubernetes)
    transport.namespace = "qps"
    transport.client = httpx.Client(
        base_url="https://kubernetes.test", transport=httpx.MockTransport(respond)
    )
    with pytest.raises(RuntimeError, match="finalizers"):
        transport.remove("trainingruns", "run-owned")
    assert [r.method for r in requests] == ["GET", "DELETE"]

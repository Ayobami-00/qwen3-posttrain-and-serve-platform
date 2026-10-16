"""Build a Forge-only API update against an immutable verified runtime build.

Usage: python tools/build_api_patch.py BASE_SOURCE_SHA
Run qps platform init first to stage the current clean, committed source archive.
"""

import json
import sys

from qps.hosted.install import Installer

s = Installer(".qps")
build = s.aws(
    "codebuild",
    "start-build",
    "--project-name",
    s.installation["bootstrap"]["build_project"],
    "--buildspec-override",
    "containers/buildspec-api.yml",
    "--environment-variables-override",
    json.dumps(
        [{"name": "QPS_API_BASE_SHA", "value": sys.argv[1], "type": "PLAINTEXT"}]
    ),
)["build"]
s.installation.setdefault("builds", []).append(
    {"id": build["id"], "operation": "images"}
)
s.save()
print("API image build started: " + build["id"], flush=True)
s.wait_build(build["id"])

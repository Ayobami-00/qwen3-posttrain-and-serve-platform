"""Explicit opt-in only: run on an already provisioned compatible NVIDIA host."""

import os
from pathlib import Path

import pytest

from qps import deploy
from qps.pipeline import coordinate
from qps.specs import absolute_spec, load_spec
from qps.store import Store


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("QPS_RUN_GPU") != "1", reason="GPU acceptance requires QPS_RUN_GPU=1"
)
def test_real_qwen3_gsm8k_recipe(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = absolute_spec(load_spec(root / "examples/local/gsm8k-lora.yaml"))
    spec.spec.artifacts.root = tmp_path / "artifacts"
    store = Store(tmp_path / "state")
    run = store.create_run(spec)
    coordinate(store, run["id"])
    assert store.output(run["id"], "training")["global_step"] == 100
    deployment = None
    try:
        deployment = deploy.create(store, run["id"], "gpu-acceptance", port=0)
        assert deploy.evaluate(store, deployment["name"])["http_offline_parity"]
    finally:
        if deployment:
            deploy.stop(store, deployment["name"])

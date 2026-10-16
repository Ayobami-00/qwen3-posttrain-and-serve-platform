import pytest

from qps.artifacts import Artifacts
from qps.hosted.agent import Agent, map_gpu_ranks


def test_rank_mapping_uses_cuda_uuid_instead_of_nvml_ordinal():
    telemetry = {"gpus": [{"index": 0, "uuid": "GPU-second"}, {"index": 1, "uuid": "GPU-first"}]}
    health = [
        {"rank": 0, "device": 0, "gpu_uuid": "GPU-first"},
        {"rank": 1, "device": 1, "gpu_uuid": "GPU-second"},
    ]
    map_gpu_ranks(telemetry, "fsdp2", "training", health)
    assert [g["rank"] for g in telemetry["gpus"]] == [1, 0]
    assert [g["cuda_device"] for g in telemetry["gpus"]] == [1, 0]
    map_gpu_ranks(telemetry, "fsdp2", "evaluation", health)
    assert all(g["rank"] is None for g in telemetry["gpus"])


def test_sync_publishes_outputs_before_concurrent_phase_references(local_run, tmp_path):
    store, run, spec = local_run
    store.update(run["id"], attempt=1)
    source = tmp_path / "files"
    source.mkdir()
    (source / "weights").write_bytes(b"fixture")
    artifacts = Artifacts(store, spec.spec.artifacts.root)
    checkpoint = artifacts.publish(run["id"], "training_checkpoint", source, {})
    agent = Agent("http://127.0.0.1:1", "allocation", "token", tmp_path / "worker")
    agent.store, agent.run_id, agent.attempt = store, run["id"], 1
    agent.provenance_emitted = True
    published, requests, adapter = set(), [], []

    def publish(record):
        published.add(record["id"])
        # Real S3 uploads take time. The training subprocess can finish and
        # register its adapter while the agent uploads an earlier checkpoint.
        if record["id"] == checkpoint and not adapter:
            adapter.append(artifacts.publish(run["id"], "lora_adapter", source, {}))
            store.emit(run["id"], 1, "training_verified", {"adapter": adapter[0]})
            store.phase(run["id"], 1, "training", "succeeded", {"adapter": adapter[0]})

    def request(method, path, **kwargs):
        requests.append(path)
        if path == "/phases":
            assert kwargs["json"]["output"]["adapter"] in published
        if path == "/events":
            payload = kwargs["json"]["events"][0]["payload"]
            if "adapter" in payload:
                assert payload["adapter"] in published

    agent.publish, agent.request = publish, request
    try:
        agent.sync()
        assert adapter and adapter[0] not in published
        assert requests == []
        agent.sync()
        assert adapter[0] in published
        assert requests == ["/events", "/events", "/phases"]
    finally:
        agent.api.close()
        agent.transfer.close()


def test_offline_probe_failure_is_retained_before_a_journal_exists(tmp_path):
    agent = Agent("http://127.0.0.1:1", "allocation", "token", tmp_path)
    (tmp_path / "compatibility.json").write_text("NCCL diagnostic\nCUDA device failure")
    sent = []
    agent.request = lambda method, path, **kwargs: sent.append((path, kwargs["json"]))
    try:
        agent.report_bootstrap_failure({"kind": "training"})
        assert sent[0][0] == "/events"
        assert sent[0][1]["events"][0]["payload"]["phase"] == "compatibility_probe"
        assert "CUDA device failure" in sent[0][1]["events"][0]["payload"]["log"]
        agent.report_bootstrap_failure({"kind": "deployment"})
        assert len(sent) == 1
    finally:
        agent.api.close()
        agent.transfer.close()


def test_probe_timeout_kills_child_and_can_still_report_its_diagnostic(
    tmp_path, monkeypatch, capsys
):
    import subprocess
    import sys

    original = subprocess.Popen
    monkeypatch.setattr(
        "qps.hosted.agent.subprocess.Popen",
        lambda command, **kwargs: original(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('probe-stage-marker', flush=True); time.sleep(60)",
            ],
            **kwargs,
        ),
    )
    agent = Agent("http://127.0.0.1:1", "allocation", "token", tmp_path)
    monkeypatch.setattr(agent, "probe_timeout_seconds", lambda workflow: 0.1)
    try:
        with pytest.raises(RuntimeError, match="deadline"):
            agent.compatibility_probe()
        assert agent.process.poll() is not None
        assert not agent.stop.is_set()
        assert "probe-stage-marker" in capsys.readouterr().out
        sent = []
        agent.request = lambda method, path, **kwargs: sent.append(kwargs["json"])
        agent.report_bootstrap_failure({"kind": "training"})
        assert "probe-stage-marker" in sent[0]["events"][0]["payload"]["log"]
    finally:
        agent.api.close()
        agent.transfer.close()


def test_probe_rank_exits_when_its_launcher_dies(tmp_path):
    import json
    import os
    import signal
    import subprocess
    import sys
    import time

    from qps.util import process_alive, signal_process

    identity_file = tmp_path / "rank.json"
    rank_code = f"from pathlib import Path; import json,time; from qps.util import process_identity; from qps.hosted.distributed_probe import start_orphan_watchdog; start_orphan_watchdog(); Path({str(identity_file)!r}).write_text(json.dumps(process_identity())); time.sleep(60)"
    launcher_code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{rank_code!r}],start_new_session=True); time.sleep(60)"
    environment = {k: v for k, v in os.environ.items() if k != "QPS_PROBE_OWNER"}
    launcher = subprocess.Popen(
        [sys.executable, "-c", launcher_code], env=environment, start_new_session=True
    )
    identity = None
    try:
        deadline = time.monotonic() + 5
        while not identity_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        identity = json.loads(identity_file.read_text())
        launcher.kill()
        launcher.wait(timeout=5)
        deadline = time.monotonic() + 3
        while process_alive(identity) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not process_alive(identity)
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=5)
        signal_process(identity, signal.SIGKILL, group=True)

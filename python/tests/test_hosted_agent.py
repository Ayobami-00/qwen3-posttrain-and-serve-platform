from qps.artifacts import Artifacts
from qps.hosted.agent import Agent


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

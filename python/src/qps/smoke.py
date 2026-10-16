"""One-command offline model training and real HTTP lifecycle proof."""

import json
import uuid
from pathlib import Path

import httpx

from qps import deploy
from qps.fixtures import create_fixture
from qps.pipeline import coordinate
from qps.store import Store
from qps.util import atomic_json


def verify_stream(client, url, body, chat=True):
    route = "/v1/chat/completions" if chat else "/v1/completions"
    response = client.post(url + route, json=body)
    response.raise_for_status()
    expected = response.json()["choices"][0]
    expected_text = expected["message"]["content"] if chat else expected["text"]
    chunks = []
    done = False
    finish = None
    with client.stream("POST", url + route, json={**body, "stream": True}) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(data["error"])
            choice = data["choices"][0]
            chunks.append(choice["delta"].get("content", "") if chat else choice["text"])
            finish = choice["finish_reason"] or finish
    if not done or "".join(chunks) != expected_text or finish != expected["finish_reason"]:
        raise ValueError("Streaming text/termination differs from normal generation")
    return {"done": done, "text": expected_text, "finish_reason": finish}


def smoke(directory: Path, verbose=True):
    directory = directory.resolve()
    name = "smoke-" + uuid.uuid4().hex[:10]
    store = Store(directory)
    fixture = directory / "fixtures" / name
    spec = create_fixture(fixture, name)
    run = store.create_run(spec)
    coordinate(store, run["id"], verbose=verbose)
    metrics = [event for event in store.events(run["id"]) if event["kind"] == "train_step"]
    if [event["payload"]["global_step"] for event in metrics] != [1, 2, 3, 4]:
        raise ValueError("Missing or duplicated optimizer-step metrics")
    deployment = None
    try:
        deployment = deploy.create(store, run["id"], name, port=0)
        evaluation = deploy.evaluate(store, name)
        with httpx.Client(trust_env=False, timeout=120) as client:
            models = client.get(deployment["url"] + "/v1/models").json()
            if models["data"][0]["artifact_id"] != deployment["artifact_id"]:
                raise ValueError("Model listing artifact identity mismatch")
            chat = verify_stream(
                client,
                deployment["url"],
                {
                    "model": name,
                    "messages": [{"role": "user", "content": "What is 2 plus 3?"}],
                    "max_tokens": 8,
                },
            )
            completion = verify_stream(
                client,
                deployment["url"],
                {
                    "model": name,
                    "prompt": "What is 2 plus 3?",
                    "max_tokens": 8,
                },
                chat=False,
            )
        artifact = deployment["artifact_id"]
        deploy.stop(store, name)
        deployment = deploy.create(store, run["id"], name, port=0)
        if deployment["artifact_id"] != artifact:
            raise ValueError("Restart changed the loaded artifact")
        deploy.evaluate(store, name)
        report = {
            "status": "passed",
            "validation": "local_cpu_only",
            "run_id": run["id"],
            "name": name,
            "artifact_id": artifact,
            "optimizer_steps": len(metrics),
            "training": store.output(run["id"], "training"),
            "parity": store.output(run["id"], "export")["parity"],
            "http_evaluation": evaluation,
            "chat_stream": chat,
            "completion_stream": completion,
            "restart_verified": True,
            "gpu_validated": False,
        }
        atomic_json(directory / f"{name}-report.json", report)
        return report
    finally:
        if deployment:
            deploy.stop(store, name)

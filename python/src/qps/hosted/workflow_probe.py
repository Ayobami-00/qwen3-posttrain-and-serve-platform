"""Download-free two-GPU acceptance of the actual TRL/DCP/export pipeline."""

import json
import sys
from pathlib import Path


def probe(directory):
    import torch

    if torch.cuda.device_count() != 2:
        raise ValueError("The workflow compatibility probe requires two CUDA devices")
    from qps.fixtures import create_fixture
    from qps.pipeline import coordinate
    from qps.store import Store

    fixture = create_fixture(directory / "fixture", "fsdp2-compatibility")
    spec = fixture.model_copy(deep=True)
    spec.spec.execution.profile = "local-cuda"
    spec.spec.execution.parallelism = "fsdp2"
    spec.spec.execution.gpu_count = 2
    spec.spec.execution.max_wall_time_seconds = 210
    spec.spec.training.dtype = "bf16"
    spec.spec.training.max_steps = 4
    spec.spec.training.gradient_accumulation_steps = 2
    spec.spec.training.checkpoint_every_steps = 2
    spec.spec.evaluation.max_new_tokens = 8
    spec.spec.artifacts.root = directory / "artifacts"
    store = Store(directory / "journal")
    run = store.create_run(spec)
    coordinate(store, run["id"], verbose=False)
    first = store.output(run["id"], "training")
    checkpoint = first["checkpoints"][0]
    coordinate(store, run["id"], action="resume", checkpoint=checkpoint, verbose=False)
    resumed = [
        e["payload"]["global_step"]
        for e in store.events(run["id"])
        if e["kind"] == "train_step" and e["attempt"] == 2
    ]
    if resumed != [3, 4]:
        raise ValueError("Distributed optimizer/data recovery did not continue at step three")
    latest = store.output(run["id"], "training")["checkpoints"][-1]
    before = sum(e["kind"] == "train_step" for e in store.events(run["id"]))
    coordinate(store, run["id"], action="resume", checkpoint=latest, verbose=False)
    after = sum(e["kind"] == "train_step" for e in store.events(run["id"]))
    if after != before or store.output(run["id"], "training")["global_step"] != 4:
        raise ValueError("Finalizing a complete checkpoint performed an extra optimizer step")
    states = [e["payload"] for e in store.events(run["id"]) if e["kind"] == "distributed_prepared"]
    if len(states) != 3 or not all(
        s["world_size"] == 2 and s["local_parameter_elements"] < s["global_parameter_elements"]
        for s in states
    ):
        raise ValueError("The actual trainer did not demonstrate two-rank parameter sharding")
    verified = [e["payload"] for e in store.events(run["id"]) if e["kind"] == "resume_verified"]
    if len(verified) != 1 or verified[0]["optimizer_min_step"] != 2:
        raise ValueError("The distributed optimizer checkpoint was not restored")
    return {
        "passed": True,
        "worldSize": 2,
        "normalPipeline": True,
        "partialCheckpointStep": 2,
        "firstResumedStep": 3,
        "lastStep": 4,
        "completedCheckpointNoExtraSteps": True,
        "parameterSharding": states,
        "reloadParity": store.output(run["id"], "export")["parity"],
        "adapterDeltaL1": store.output(run["id"], "training")["adapter_delta_l1"],
    }


if __name__ == "__main__":
    print(json.dumps(probe(Path(sys.argv[1]).resolve())), flush=True)

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from qps.pipeline import coordinate, reconcile
from qps.util import file_lock, process_alive, process_identity


def test_duplicate_coordinator_is_rejected(local_run):
    store, run, _ = local_run
    with file_lock(store.directory(run["id"]) / "coordinator.lock"):
        with pytest.raises(RuntimeError, match="already owns"):
            coordinate(store, run["id"], verbose=False)


def test_timeout_kills_the_worker_and_records_failure(local_run, monkeypatch):
    store, run, spec = local_run
    spec.spec.execution.max_wall_time_seconds = 1
    with store.connect() as db:
        db.execute(
            "UPDATE runs SET spec=? WHERE id=?", (spec.model_dump_json(by_alias=True), run["id"])
        )
    original = subprocess.Popen
    children = []

    def slow_worker(command, **kwargs):
        process = original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr("qps.pipeline.subprocess.Popen", slow_worker)
    with pytest.raises(TimeoutError):
        coordinate(store, run["id"], verbose=False)
    assert children[0].poll() is not None
    assert store.run(run["id"])["status"] == "failed"
    assert any(e["kind"] == "phase_failed" for e in store.events(run["id"]))


def test_dead_coordinator_reconciliation_and_pid_reuse_safety(local_run):
    store, run, _ = local_run
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    identity = process_identity(child.pid)
    try:
        # The current PID with a different birth time represents a reused PID, not an owner.
        owner = {"pid": os.getpid(), "created": 0}
        attempt = store.begin_attempt(run["id"], owner)
        store.phase(run["id"], attempt, "training", "running")
        store.update(run["id"], worker=json.dumps(identity))
        assert reconcile(store, run["name"])["status"] == "interrupted"
        child.wait(timeout=5)
        assert not process_alive(identity)
        assert os.getpid() == owner["pid"]  # Reconciliation never signalled the reused owner PID.
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_background_cancellation_and_orphan_watchdog(local_run):
    from qps.pipeline import background

    store, run, _ = local_run
    owner = background(store, run["id"], action="run")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = store.run(run["id"])
        if record["worker"]:
            break
        time.sleep(0.02)
    else:
        pytest.fail("Coordinator did not launch a worker")
    worker = json.loads(record["worker"])
    # Abrupt coordinator death must not leave the worker alive indefinitely.
    os.kill(owner["pid"], signal.SIGKILL)
    deadline = time.monotonic() + 5
    while process_alive(worker) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not process_alive(worker)
    assert reconcile(store, run["id"])["status"] == "interrupted"


def test_cooperative_cancel(local_run, monkeypatch):
    import threading

    store, run, _ = local_run
    original = subprocess.Popen
    child = []

    def slow_worker(command, **kwargs):
        process = original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        child.append(process)
        store.update(run["id"], cancel=1)
        return process

    monkeypatch.setattr("qps.pipeline.subprocess.Popen", slow_worker)
    assert threading.current_thread() is threading.main_thread()
    with pytest.raises(InterruptedError):
        coordinate(store, run["id"], verbose=False)
    assert store.run(run["id"])["status"] == "cancelled"
    assert child[0].poll() is not None

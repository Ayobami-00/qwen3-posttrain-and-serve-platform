"""Sequential, restartable process orchestration; this module does not import torch."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from qps.artifacts import Artifacts, verify_artifact
from qps.specs import RunSpec
from qps.store import Store
from qps.util import canonical, file_lock, process_alive, process_identity, signal_process

PHASES = ("prepare", "baseline", "training", "export", "evaluation")
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def format_event(event):
    payload = event["payload"]
    prefix = f"{event['time']} attempt={event['attempt']} {event['kind']}"
    if event["kind"] == "train_step":
        return (
            f"{prefix} step={payload['global_step']} loss={payload['loss']:.6f} "
            f"lr={payload['learning_rate']:.3g} grad_norm={payload['grad_norm']} "
            f"tokens={payload['tokens_seen']} elapsed={payload['elapsed_seconds']:.2f}s"
        )
    return prefix + " " + canonical(payload)


def reconcile(store: Store, run_id: str):
    """Never signal a reused PID. Only a dead coordinator's worker may be reclaimed."""
    run = store.run(run_id)
    run_id = run["id"]
    owner = json.loads(run["owner"]) if run["owner"] else None
    if run["status"] == "running" and not process_alive(owner):
        worker = json.loads(run["worker"]) if run["worker"] else None
        signal_process(worker, signal.SIGTERM, group=True)
        deadline = time.monotonic() + 5
        while process_alive(worker) and time.monotonic() < deadline:
            time.sleep(0.1)
        signal_process(worker, signal.SIGKILL, group=True)
        store.phase(run_id, run["attempt"], run["phase"], "failed")
        store.finish(run_id, run["attempt"], "interrupted", "Coordinator exited before completing")
    return store.run(run_id)


def resolve_checkpoint(store: Store, run_id: str, value: str | None):
    records = [
        record for record in store.artifacts(run_id) if record["kind"] == "training_checkpoint"
    ]
    if not records:
        raise ValueError("No complete registered checkpoint is available")
    if value and value != "latest":
        matches = [
            r
            for r in records
            if r["id"] == value
            or Path(value).resolve()
            in (
                Path(r["path"]),
                Path(r["path"]) / "files",
            )
        ]
        if not matches:
            raise ValueError(
                "Checkpoint must be a registered artifact ID or its verified directory"
            )
        record = matches[0]
    else:
        record = max(
            records,
            key=lambda r: (
                verify_artifact(store, r["id"])[1]["metadata"]["global_step"],
                r["created"],
            ),
        )
    verify_artifact(store, record["id"])
    return record["id"]


def verify_output(store, output):
    if isinstance(output, dict):
        for value in output.values():
            verify_output(store, value)
    elif isinstance(output, list):
        for value in output:
            verify_output(store, value)
    elif isinstance(output, str) and output.startswith("art-"):
        verify_artifact(store, output)


def stop_worker(process, identity):
    signal_process(identity, signal.SIGTERM, group=True)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        signal_process(identity, signal.SIGKILL, group=True)
        process.wait(timeout=5)


def coordinate(store: Store, run_id: str, action="run", checkpoint=None, verbose=True):
    run_id = store.run(run_id)["id"]
    directory = store.directory(run_id)
    with file_lock(directory / "coordinator.lock", blocking=False):
        run = reconcile(store, run_id)
        if run["status"] == "running":
            raise ValueError("Run is already active")
        if action == "run" and run["status"] != "pending":
            raise ValueError("Use runs resume or runs retry for an existing run")
        if action != "run" and run["status"] not in TERMINAL:
            raise ValueError("Recovery requires a terminal or interrupted run")
        spec = RunSpec.model_validate_json(run["spec"])
        Artifacts(store, spec.spec.artifacts.root).reconcile()
        Artifacts(store, store.root / "datasets").reconcile()
        remaining = spec.spec.execution.max_wall_time_seconds - run["elapsed_seconds"]
        if remaining <= 0:
            raise ValueError(
                "This run exhausted its total active wall-time budget; submit a new run"
            )
        if action == "retry-evaluation":
            for phase in PHASES[:-1]:
                output = store.output(run_id, phase)
                if output is None:
                    raise ValueError(f"Evaluation retry requires completed phase: {phase}")
                verify_output(store, output)
        resume_id = None
        if action == "resume" and (checkpoint or not store.output(run_id, "training")):
            records = [r for r in store.artifacts(run_id) if r["kind"] == "training_checkpoint"]
            if records or checkpoint:
                resume_id = resolve_checkpoint(store, run_id, checkpoint)
        invalidated = (
            PHASES[2:] if resume_id else ("evaluation",) if action == "retry-evaluation" else ()
        )
        if invalidated:
            with store.connect() as db:
                for phase in invalidated:
                    db.execute(
                        "UPDATE phases SET status='superseded' WHERE run_id=? AND phase=? "
                        "AND status='succeeded'",
                        (run_id, phase),
                    )
        owner = process_identity()
        attempt = store.begin_attempt(run_id, owner)
        started = time.monotonic()
        deadline = started + remaining
        interrupted = False

        def cancel_handler(signum, frame):
            nonlocal interrupted
            interrupted = True

        old_handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, cancel_handler)
        process = identity = None
        cursor = 0
        current_phase = None
        try:
            for phase in PHASES:
                redo = action == "retry-evaluation" and phase == "evaluation"
                redo = redo or (
                    resume_id is not None and phase in ("training", "export", "evaluation")
                )
                previous = store.output(run_id, phase)
                if previous and not redo:
                    verify_output(store, previous)
                    continue
                current_phase = phase
                store.phase(run_id, attempt, phase, "running")
                work = directory / "attempts" / str(attempt)
                work.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "-m",
                    "qps.worker",
                    "--state",
                    str(store.root),
                    "--run",
                    run_id,
                    "--attempt",
                    str(attempt),
                    "--phase",
                    phase,
                    "--owner",
                    canonical(owner),
                ]
                if phase == "training" and resume_id:
                    command += ["--checkpoint", resume_id]
                environment = {
                    **os.environ,
                    "TOKENIZERS_PARALLELISM": "false",
                    "OMP_NUM_THREADS": str(spec.spec.execution.threads),
                }
                with (work / f"{phase}.log").open("a", buffering=1) as log:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=environment,
                    )
                    identity = process_identity(process.pid)
                    store.update(run_id, worker=canonical(identity))
                    while True:
                        events = store.events(run_id, cursor)
                        for event in events:
                            cursor = event["id"]
                            if verbose:
                                print(format_event(event), flush=True)
                        elapsed = time.monotonic() - started
                        store.update(run_id, elapsed_seconds=run["elapsed_seconds"] + elapsed)
                        if interrupted or store.run(run_id)["cancel"]:
                            raise InterruptedError(
                                "Run cancelled; resume from the last complete checkpoint"
                            )
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Total active wall-time limit exceeded")
                        result = process.poll()
                        if result is not None:
                            break
                        time.sleep(0.1)
                    if result:
                        tail = (work / f"{phase}.log").read_text(errors="replace")[-6000:]
                        raise RuntimeError(
                            f"{phase} worker exited {result}; log: {work / f'{phase}.log'}\n{tail}"
                        )
                output = json.loads((work / f"{phase}-output.json").read_text())
                verify_output(store, output)
                store.phase(run_id, attempt, phase, "succeeded", output)
                store.update(run_id, worker=None)
                process = identity = None
            store.finish(run_id, attempt, "succeeded")
        except BaseException as exc:
            if process is not None and process.poll() is None:
                stop_worker(process, identity)
            if current_phase:
                store.phase(run_id, attempt, current_phase, "failed")
            status = (
                "cancelled" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed"
            )
            store.finish(run_id, attempt, status, str(exc))
            raise
        finally:
            store.update(
                run_id, elapsed_seconds=run["elapsed_seconds"] + time.monotonic() - started
            )
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            with contextlib.suppress(Exception):
                store.project(run_id)
            if verbose:
                for event in store.events(run_id, cursor):
                    print(format_event(event), flush=True)
        return store.run(run_id)


def background(store, run_id, action="run", checkpoint=None):
    directory = store.directory(run_id)
    command = [sys.executable, "-m", "qps.pipeline", str(store.root), run_id, action]
    if checkpoint:
        command += [checkpoint]
    with (directory / "coordinator.log").open("a") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    return process_identity(process.pid)


if __name__ == "__main__":
    coordinate(
        Store(Path(sys.argv[1])),
        sys.argv[2],
        sys.argv[3],
        sys.argv[4] if len(sys.argv) > 4 else None,
    )

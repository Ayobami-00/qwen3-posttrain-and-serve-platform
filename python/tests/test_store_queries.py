"""Run-scoped cursors and committed phase outputs remain clear across attempts."""

import json

import pytest

from qps.util import process_identity


def test_event_cursors_are_exclusive_and_do_not_leak_other_runs(local_run):
    store, run, spec = local_run
    first = store.begin_attempt(run["id"], process_identity())
    store.emit(run["id"], first, "train_step", {"global_step": 1})
    cursor = store.events(run["id"])[-1]["id"]
    second_spec = spec.model_copy(deep=True)
    second_spec.metadata.name = "other-run"
    other = store.create_run(second_spec)
    second = store.begin_attempt(other["id"], process_identity())
    store.emit(other["id"], second, "train_step", {"global_step": 99})

    assert store.events(run["id"], cursor) == []
    store.emit(run["id"], first, "train_step", {"global_step": 2})
    page = store.events(run["id"], cursor)

    assert len(page) == 1
    assert page[0]["id"] > cursor
    assert page[0]["payload"] == {"global_step": 2}
    assert store.events(run["id"], page[-1]["id"]) == []


def test_superseded_attempt_cannot_append_to_the_new_attempt(local_run):
    store, run, _ = local_run
    first = store.begin_attempt(run["id"], process_identity())
    store.emit(run["id"], first, "train_step", {"global_step": 1})
    store.finish(run["id"], first, "interrupted")
    second = store.begin_attempt(run["id"], process_identity())
    before = store.events(run["id"])

    with pytest.raises(RuntimeError, match="Stale attempt"):
        store.emit(run["id"], first, "train_step", {"global_step": 2})
    assert store.events(run["id"]) == before
    store.emit(run["id"], second, "train_step", {"global_step": 2})
    metrics = store.directory(run["id"]) / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [(row["attempt"], row["payload"]["global_step"]) for row in rows] == [
        (first, 1),
        (second, 2),
    ]


def test_only_successful_phase_outputs_replace_the_last_committed_output(local_run):
    store, run, _ = local_run
    first = store.begin_attempt(run["id"], process_identity())
    previous = {"dataset": "previous"}
    store.phase(run["id"], first, "prepare", "succeeded", previous)
    store.finish(run["id"], first, "interrupted")
    second = store.begin_attempt(run["id"], process_identity())

    store.phase(run["id"], second, "prepare", "running")
    assert store.output(run["id"], "prepare") == previous
    store.phase(run["id"], second, "prepare", "failed", {"dataset": "incomplete"})
    assert store.output(run["id"], "prepare") == previous
    store.phase(run["id"], second, "prepare", "succeeded", {"dataset": "replacement"})
    assert store.output(run["id"], "prepare") == {"dataset": "replacement"}
    assert store.output(run["id"], "export") is None

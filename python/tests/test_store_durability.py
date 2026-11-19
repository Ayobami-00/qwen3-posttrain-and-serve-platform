"""SQLite events survive failed projections and invalid metrics never reach either sink."""

import json

import pytest

from qps.util import process_identity


def test_event_commit_survives_an_interrupted_jsonl_projection(local_run, monkeypatch):
    store, run, _ = local_run
    attempt = store.begin_attempt(run["id"], process_identity())
    cursor = store.events(run["id"])[-1]["id"]

    def unavailable(run_id):
        raise OSError("projection destination unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(store, "project", unavailable)
        with pytest.raises(OSError, match="projection destination unavailable"):
            store.emit(run["id"], attempt, "train_step", {"global_step": 1, "loss": 0.5})

    committed = store.events(run["id"], cursor)
    assert len(committed) == 1
    assert committed[0]["payload"] == {"global_step": 1, "loss": 0.5}
    store.project(run["id"])
    metrics = store.directory(run["id"]) / "metrics.jsonl"
    assert [json.loads(line) for line in metrics.read_text().splitlines()] == committed


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_nested_nonfinite_metrics_do_not_change_events_or_projections(local_run, number):
    store, run, _ = local_run
    attempt = store.begin_attempt(run["id"], process_identity())
    before = store.events(run["id"])
    events_path = store.directory(run["id"]) / "events.jsonl"
    metrics_path = store.directory(run["id"]) / "metrics.jsonl"
    projections = events_path.read_bytes(), metrics_path.read_bytes()

    with pytest.raises(ValueError):
        store.emit(run["id"], attempt, "train_step", {"ranks": [{"loss": number}]})

    assert store.events(run["id"]) == before
    assert (events_path.read_bytes(), metrics_path.read_bytes()) == projections


def test_missing_jsonl_projections_are_rebuilt_from_the_journal(local_run):
    store, run, _ = local_run
    attempt = store.begin_attempt(run["id"], process_identity())
    store.emit(run["id"], attempt, "train_step", {"global_step": 1, "loss": 0.5})
    directory = store.directory(run["id"])
    expected = {name: (directory / name).read_bytes() for name in ("events.jsonl", "metrics.jsonl")}
    for name in expected:
        (directory / name).unlink()

    store.project(run["id"])

    assert {name: (directory / name).read_bytes() for name in expected} == expected

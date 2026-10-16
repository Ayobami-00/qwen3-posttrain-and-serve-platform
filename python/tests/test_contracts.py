import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from qps.artifacts import Artifacts, file_manifest, verify_artifact
from qps.data import encode, final_answer, normalize, prepare_records
from qps.specs import DatasetSpec, RunSpec, load_spec
from qps.util import process_identity


def test_cli_import_is_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import qps.cli; "
                "assert all(name not in sys.modules for name in "
                "('torch','transformers','trl','peft','datasets'))"
            ),
        ],
        check=True,
    )


def test_examples_and_schema_are_current():
    root = Path(__file__).resolve().parents[2]
    for name in ("tiny-cpu", "gsm8k-lora"):
        load_spec(root / "examples" / "local" / f"{name}.yaml")
    assert json.loads(
        (root / "contracts" / "run.schema.json").read_text()
    ) == RunSpec.model_json_schema(by_alias=True)


def test_profile_mismatch_and_unknown_settings_fail(local_run):
    _, _, spec = local_run
    data = spec.model_dump(mode="json", by_alias=True)
    data["spec"]["training"]["dtype"] = "bf16"
    with pytest.raises(ValidationError, match="requires fp32"):
        RunSpec.model_validate(data)
    data["spec"]["training"]["dtype"] = "fp32"
    data["spec"]["training"]["mispelledSteps"] = 100
    with pytest.raises(ValidationError, match="Extra inputs"):
        RunSpec.model_validate(data)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work\n#### 1,234.50", "2469/2"),
        ("#### -1/2", "-1/2"),
        ("#### +0.25", "1/4"),
        ("#### .5", "1/2"),
        ("#### 0", "0"),
        ("no marker 3", None),
        ("#### 1,2", None),
        ("#### 1/0", None),
        ("#### nan", None),
        ("#### __import__('os')", None),
        ("#### 2 extra", None),
        ("#### 1\n#### 2", "2"),
    ],
)
def test_numeric_answer_contract(text, expected):
    result = final_answer(text)
    assert (str(result) if result is not None else None) == expected


class Tokenizer:
    eos_token_id = 3

    def apply_chat_template(self, *args, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [1, 2]

    def encode(self, text, **kwargs):
        return [4 + ord(character) % 20 for character in text]


def test_grouped_splits_are_stable_and_complete_answers_not_truncated():
    rows = [
        {"question": f"Question {number}", "answer": f"work #### {number}"} for number in range(6)
    ]
    rows += [{"question": " Question   0 ", "answer": "alternate #### 0"}]
    rows += [{"question": "proof", "answer": "no verifiable answer"}]
    spec = DatasetSpec(id="local", train_examples=3, validation_examples=2)
    first = prepare_records(rows, Tokenizer(), spec, 128)
    second = prepare_records(reversed(rows), Tokenizer(), spec, 128)
    assert first[:2] == second[:2]
    assert set(first[2]["train_ids"]).isdisjoint(first[2]["validation_ids"])
    assert first[2]["duplicates"] == 1 and len(first[2]["rejected"]) == 1
    row = encode(Tokenizer(), normalize(rows[0]), 128)
    assert row["labels"][:2] == [-100, -100]
    assert row["labels"][-1] == Tokenizer.eos_token_id
    with pytest.raises(ValueError, match="token budget"):
        encode(Tokenizer(), normalize(rows[0]), 4)


def test_padding_does_not_mask_a_real_eos():
    from qps.ml import collate

    batch = collate(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [1, 3], "labels": [-100, 3]},
        ],
        pad_token_id=3,
    )
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 3, -100]]


def test_artifact_integrity_and_rename_commit_recovery(local_run, tmp_path, monkeypatch):
    store, run, spec = local_run
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights").write_bytes(b"original")
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "weights").write_bytes(b"original")
    assert file_manifest(source) == file_manifest(copy)
    artifacts = Artifacts(store, spec.spec.artifacts.root)
    original = artifacts._register

    def interrupted(*args):
        raise RuntimeError("database connection lost after rename")

    monkeypatch.setattr(artifacts, "_register", interrupted)
    with pytest.raises(RuntimeError):
        artifacts.publish(run["id"], "test", source, {})
    monkeypatch.setattr(artifacts, "_register", original)
    artifacts.reconcile()
    records = store.artifacts(run["id"])
    assert len(records) == 1
    files, _ = verify_artifact(store, records[0]["id"])
    (files / "weights").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        verify_artifact(store, records[0]["id"])


def test_events_are_fenced_finite_and_rebuildable(local_run):
    store, run, _ = local_run
    attempt = store.begin_attempt(run["id"], process_identity())
    store.emit(run["id"], attempt, "train_step", {"global_step": 1, "loss": 0.5})
    with pytest.raises(RuntimeError, match="Stale"):
        store.emit(run["id"], 0, "train_step", {})
    with pytest.raises(ValueError):
        store.emit(run["id"], attempt, "train_step", {"loss": float("nan")})
    metrics = store.directory(run["id"]) / "metrics.jsonl"
    original = metrics.read_text()
    metrics.write_text("interrupted file projection")
    store.project(run["id"])
    assert metrics.read_text() == original
    with pytest.raises(ValueError, match="already exists"):
        store.create_run(RunSpec.model_validate_json(run["spec"]))

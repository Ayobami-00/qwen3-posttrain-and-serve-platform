"""Prepared membership and rejection reports are reproducible source evidence."""

import pytest

from qps.data import normalize, prepare_records
from qps.specs import DatasetSpec


def rows(count):
    return [
        {"question": f"Question {index}", "answer": f"z explanation #### {index}"}
        for index in range(count)
    ]


def test_split_membership_and_duplicate_solution_selection_ignore_source_order(
    character_tokenizer,
):
    source = rows(8)
    source.append({"question": " Question   0 ", "answer": "a explanation #### 0"})
    spec = DatasetSpec(id="fixture", source="local", train_examples=4, validation_examples=4)

    first = prepare_records(source, character_tokenizer, spec, 64)
    second = prepare_records(reversed(source), character_tokenizer, spec, 64)

    assert first[:2] == second[:2]
    train, validation, report = first
    assert len(train) == len(validation) == 4
    assert set(report["train_ids"]).isdisjoint(report["validation_ids"])
    assert len(set(report["train_ids"] + report["validation_ids"])) == 8
    identity = normalize(source[-1])["problem_id"]
    selected = next(row for row in train + validation if row["problem_id"] == identity)
    assert selected["solution"] == "a explanation #### 0"
    assert report["duplicates"] == 1
    assert report["eligible_groups"] == 8
    assert report["rejected"] == []


def test_rejection_report_retains_source_offsets_and_reasons(character_tokenizer):
    source = rows(5)
    source += [
        {"question": "Missing marker", "answer": "not verifiable"},
        {"answer": "#### 1"},
        None,
        {"question": "Too long", "answer": "x" * 100 + " #### 1"},
    ]
    spec = DatasetSpec(id="fixture", source="local", train_examples=3, validation_examples=2)
    train, validation, report = prepare_records(source, character_tokenizer, spec, 64)

    assert len(train) == 3 and len(validation) == 2
    assert [row["row"] for row in report["rejected"]] == [5, 6, 7, 8]
    assert all(row["reason"] for row in report["rejected"])
    assert "token budget" in report["rejected"][-1]["reason"]
    assert report["eligible_groups"] == 5


def test_filtering_and_duplicates_cannot_satisfy_the_unique_example_budget(character_tokenizer):
    source = rows(1) * 4 + [{"question": "Invalid", "answer": "unmarked"}]
    spec = DatasetSpec(id="fixture", source="local", train_examples=3, validation_examples=2)

    with pytest.raises(ValueError, match="Need 5 valid unique problems; found 1"):
        prepare_records(source, character_tokenizer, spec, 64)

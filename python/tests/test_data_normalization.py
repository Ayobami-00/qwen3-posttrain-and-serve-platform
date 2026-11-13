"""Question grouping must survive source formatting without rewriting prompts."""

import copy

import pytest

from qps.data import normalize


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("What is 1 + 2?", "  What\t is   1 + 2?\n"),
        ("What is 1 + 2?", "Ｗｈａｔ ｉｓ １ ＋ ２？"),
        ("Café costs 3", "Cafe\u0301 costs 3"),
        ("How many apples?", "How\u00a0many apples?"),
    ],
)
def test_equivalent_question_formatting_has_one_problem_identity(first, second):
    left = normalize({"question": first, "answer": "solution #### 3"})
    right = normalize({"question": second, "answer": "solution #### 3"})
    assert left["problem_id"] == right["problem_id"]
    assert len(left["problem_id"]) == 64
    assert left["question"] == first.strip()
    assert right["question"] == second.strip()


def test_normalization_removes_calculator_annotations_and_preserves_input():
    row = {"question": "  Compute 1 + 2.  ", "answer": "Compute <<1+2=3>>3.\n#### 3"}
    original = copy.deepcopy(row)
    result = normalize(row)

    assert row == original
    assert result["question"] == "Compute 1 + 2."
    assert result["solution"] == "Compute 3.\n#### 3"
    assert result["reference_answer"] == "3"


def test_solution_choice_does_not_change_question_identity():
    first = normalize({"question": "What is 2 + 2?", "answer": "add #### 4"})
    second = normalize({"question": "What is 2 + 2?", "answer": "count #### 4.0"})
    other = normalize({"question": "What is 2 + 3?", "answer": "add #### 5"})

    assert first["problem_id"] == second["problem_id"]
    assert first["reference_answer"] == second["reference_answer"]
    assert first["problem_id"] != other["problem_id"]


@pytest.mark.parametrize(
    "row",
    [
        {"question": " \t\n", "answer": "#### 1"},
        {"question": "Question", "answer": "unmarked 1"},
        {"question": "Question", "answer": "#### 1 unit"},
    ],
)
def test_unverifiable_rows_are_rejected_before_tokenization(row):
    with pytest.raises(ValueError, match="Missing question or verifiable final numeric answer"):
        normalize(row)

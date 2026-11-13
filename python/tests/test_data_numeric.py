"""Exact numeric comparison and bounded parsing at the evaluation boundary."""

from fractions import Fraction

import pytest

from qps.data import final_answer, numeric


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  +0001  ", Fraction(1)),
        ("-.125", Fraction(-1, 8)),
        ("1,000,000", Fraction(1_000_000)),
        ("12,345.5", Fraction(24_691, 2)),
        ("6/8", Fraction(3, 4)),
        ("-0", Fraction(0)),
    ],
)
def test_numeric_answers_preserve_exact_fraction_values(text, expected):
    result = numeric(text)
    assert isinstance(result, Fraction)
    assert result == expected


@pytest.mark.parametrize(
    "text",
    ["", "1e3", "1_000", "3/0", "1,23", "1,000,00", "1 000", "5 dollars", "(3)", "inf"],
)
def test_numeric_answers_reject_ambiguous_or_non_numeric_suffixes(text):
    assert numeric(text) is None


def test_numeric_length_budget_includes_the_complete_number():
    maximum = "1" * 128
    assert numeric(maximum) == Fraction(maximum)
    assert numeric(maximum + "1") is None


def test_equivalent_decimal_and_fraction_answers_compare_exactly():
    assert final_answer("decimal #### 0.1") == final_answer("fraction #### 1/10")
    assert numeric("0.1") + numeric("0.2") == numeric("0.3")


def test_last_answer_marker_does_not_fall_back_to_an_earlier_valid_answer():
    assert final_answer("draft #### 4\ncorrected #### not-a-number") is None
    assert final_answer("draft #### 4\ncorrected #### -3/2") == Fraction(-3, 2)

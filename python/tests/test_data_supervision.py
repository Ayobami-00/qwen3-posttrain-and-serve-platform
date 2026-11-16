"""Explicit labels preserve complete assistant responses and prompt settings."""

import copy

import pytest

from qps.data import INSTRUCTION, encode, normalize


def test_encoding_shares_the_non_thinking_chat_profile(character_tokenizer):
    record = normalize({"question": "What is 1 + 2?", "answer": "Add the values.\n#### 3"})
    encode(character_tokenizer, record, 128)

    assert character_tokenizer.messages == [
        {"role": "user", "content": "What is 1 + 2?\n" + INSTRUCTION}
    ]
    assert character_tokenizer.template_options == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert character_tokenizer.response == record["solution"]
    assert character_tokenizer.encode_options == {"add_special_tokens": False}


def test_only_prompt_tokens_are_ignored_and_the_complete_solution_is_supervised(
    character_tokenizer,
):
    record = normalize({"question": "Compute", "answer": "First reason.\n#### 3"})
    original = copy.deepcopy(record)
    result = encode(character_tokenizer, record, 128)
    response = [1000 + ord(c) for c in record["solution"]] + [99]

    assert record == original
    assert result["prompt_ids"] == [10, 11]
    assert result["input_ids"] == [10, 11] + response
    assert result["labels"] == [-100, -100] + response
    assert len(result["input_ids"]) == len(result["labels"])
    assert result["labels"][-1] == character_tokenizer.eos_token_id
    assert result["reference_answer"] == "3"


def test_sequence_budget_accepts_exact_fit_including_eos(character_tokenizer):
    record = normalize({"question": "Compute", "answer": "#### 3"})
    complete_length = 2 + len(record["solution"]) + 1
    result = encode(character_tokenizer, record, complete_length)

    assert len(result["input_ids"]) == complete_length
    with pytest.raises(ValueError, match="Complete example exceeds the token budget"):
        encode(character_tokenizer, record, complete_length - 1)

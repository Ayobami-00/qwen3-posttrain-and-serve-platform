"""Source normalization, deterministic splits, prompt rendering and explicit labels."""

import hashlib
import re
import unicodedata
from fractions import Fraction

INSTRUCTION = "Show your solution and end with #### followed by the final numeric answer."
NUMBER = re.compile(r"[+-]?(?:\d+/\d+|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)")


def numeric(value: str) -> Fraction | None:
    value = value.strip()
    if len(value) > 128 or not NUMBER.fullmatch(value):
        return None
    try:
        return Fraction(value.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def final_answer(text: str) -> Fraction | None:
    _, marker, answer = text.rpartition("####")
    return numeric(answer) if marker else None


def normalize(row: dict) -> dict:
    question = row["question"].strip()
    solution = re.sub(r"<<[^<>]*>>", "", row["answer"]).strip()
    reference = final_answer(solution)
    if not question or reference is None:
        raise ValueError("Missing question or verifiable final numeric answer")
    canonical_question = unicodedata.normalize("NFKC", " ".join(question.split()))
    return {
        "problem_id": hashlib.sha256(canonical_question.encode()).hexdigest(),
        "question": question,
        "solution": solution,
        "reference_answer": str(reference),
    }


def prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def question_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": question + "\n" + INSTRUCTION}]


def encode(tokenizer, record: dict, maximum: int) -> dict:
    prefix = prompt_ids(tokenizer, question_messages(record["question"]))
    response = tokenizer.encode(record["solution"], add_special_tokens=False)
    response.append(tokenizer.eos_token_id)
    if len(prefix) + len(response) > maximum:
        raise ValueError("Complete example exceeds the token budget")
    return {
        **record,
        "prompt_ids": prefix,
        "input_ids": prefix + response,
        "labels": [-100] * len(prefix) + response,
    }


def prepare_records(rows, tokenizer, dataset_spec, maximum: int):
    groups = {}
    rejected = []
    duplicates = 0
    for index, raw in enumerate(rows):
        try:
            record = encode(tokenizer, normalize(raw), maximum)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"row": index, "reason": str(exc)})
            continue
        problem_id = record["problem_id"]
        if problem_id in groups:
            duplicates += 1
            # A stable choice among alternate solutions, independent of input order.
            if record["solution"] >= groups[problem_id]["solution"]:
                continue
        groups[problem_id] = record

    def rank(problem_id):
        value = f"gsm8k-split-v1:{dataset_spec.split_seed}:{problem_id}"
        return hashlib.sha256(value.encode()).digest()

    ordered = [groups[key] for key in sorted(groups, key=rank)]
    train_count = dataset_spec.train_examples
    total = train_count + dataset_spec.validation_examples
    if len(ordered) < total:
        raise ValueError(f"Need {total} valid unique problems; found {len(ordered)}")
    train, validation = ordered[:train_count], ordered[train_count:total]
    return (
        train,
        validation,
        {
            "normalizer": "gsm8k-v1",
            "split_policy": "sha256-groups-v1",
            "split_seed": dataset_spec.split_seed,
            "duplicates": duplicates,
            "rejected": rejected,
            "eligible_groups": len(ordered),
            "train_ids": [row["problem_id"] for row in train],
            "validation_ids": [row["problem_id"] for row in validation],
        },
    )

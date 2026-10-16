"""Generate a tiny real Qwen3 checkpoint and tokenizer entirely offline."""

import json
from pathlib import Path

from qps.specs import RunSpec
from qps.util import atomic_json

TEMPLATE = """{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{% endif %}"""


def create_fixture(directory: Path, name: str) -> RunSpec:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, set_seed

    torch.set_num_threads(2)
    set_seed(42)
    model_dir = directory / "parent"
    model_dir.mkdir(parents=True)
    rows = [
        {"question": f"What is {a} plus {b}?", "answer": f"{a} + {b} = {a + b}.\n#### {a + b}"}
        for a in range(1, 10)
        for b in range(1, 10)
    ]
    corpus = [json.dumps(row) for row in rows] + [
        "user assistant system Show your solution and end with #### followed by the final numeric answer."
    ]
    engine = Tokenizer(models.BPE(unk_token="[UNK]"))
    engine.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    engine.decoder = decoders.ByteLevel()
    engine.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=384,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", "<think>", "</think>"],
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=engine,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        model_max_length=512,
        additional_special_tokens=["<|im_start|>", "<think>", "</think>"],
    )
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(model_dir)
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=None,
    )
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(model_dir, safe_serialization=True)
    data = directory / "data.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value = {
        "metadata": {"name": name},
        "spec": {
            "model": {"id": str(model_dir.resolve()), "source": "local"},
            "dataset": {
                "id": str(data.resolve()),
                "source": "local",
                "trainExamples": 8,
                "validationExamples": 4,
            },
            "training": {
                "dtype": "fp32",
                "maxSteps": 4,
                "maxSequenceLength": 192,
                "gradientAccumulationSteps": 2,
                "checkpointEverySteps": 2,
                "learningRate": 0.001,
                "lora": {"rank": 2, "alpha": 4, "dropout": 0},
            },
            "evaluation": {"maxNewTokens": 8},
            "execution": {"profile": "local-cpu-smoke", "gpuCount": 0, "maxWallTimeSeconds": 300},
            "artifacts": {"root": str((directory.parent / "artifacts").resolve())},
        },
    }
    spec = RunSpec.model_validate(value)
    atomic_json(directory / "fixture-spec.json", spec.model_dump(mode="json", by_alias=True))
    return spec

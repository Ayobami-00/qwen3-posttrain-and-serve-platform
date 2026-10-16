"""Shared ML primitives. Imported only by worker/server processes, never CLI validation."""

import os
import threading

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria


def configure(spec):
    torch.set_num_threads(spec.execution.threads)
    if spec.execution.profile == "local-cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("local-cuda requires an NVIDIA CUDA GPU with BF16 support")
        torch.backends.cuda.matmul.allow_tf32 = False
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


def tokenizer_from(directory):
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token != "<|im_end|>" or not tokenizer.chat_template:
        raise ValueError("Qwen3 non-thinking profile requires a chat template and <|im_end|> EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(directory, spec, *, dtype_override=None):
    device, dtype = configure(spec)
    if dtype_override is not None:
        dtype = dtype_override
    model = AutoModelForCausalLM.from_pretrained(
        directory,
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
    ).to(device)
    if model.config.model_type != "qwen3":
        raise ValueError("Only dense Qwen3 models are supported")
    return model


def collate(rows, pad_token_id):
    ids = [torch.tensor(row["input_ids"], dtype=torch.long) for row in rows]
    labels = [torch.tensor(row["labels"], dtype=torch.long) for row in rows]
    return {
        "input_ids": pad_sequence(ids, batch_first=True, padding_value=pad_token_id),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        "attention_mask": pad_sequence(
            [torch.ones_like(sequence) for sequence in ids],
            batch_first=True,
            padding_value=0,
        ),
    }


class CancelGeneration(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device
        )


@torch.inference_mode()
def generate(model, tokenizer, input_ids, maximum, cancel=None, streamer=None):
    if not input_ids:
        raise ValueError("Empty prompt")
    if len(input_ids) + maximum > model.config.max_position_embeddings:
        raise ValueError("Prompt plus requested output exceeds the model context limit")
    model.eval()
    tensor = torch.tensor([input_ids], device=model.device)
    output = model.generate(
        input_ids=tensor,
        attention_mask=torch.ones_like(tensor),
        max_new_tokens=maximum,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        temperature=None,
        top_p=None,
        top_k=None,
        stopping_criteria=[CancelGeneration(cancel)] if cancel else None,
        streamer=streamer,
    )
    ids = output[0, len(input_ids) :].tolist()
    return {
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "token_ids": ids,
        "finish_reason": "stop" if ids and ids[-1] == tokenizer.eos_token_id else "length",
        "prompt_tokens": len(input_ids),
        "completion_tokens": len(ids),
    }


@torch.inference_mode()
def probe_logits(model, input_ids):
    model.eval()
    tensor = torch.tensor([input_ids[:24]], device=model.device)
    return (
        model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False)
        .logits.float()
        .cpu()
    )


PARITY_TOLERANCES = {
    # Qwen3-0.6B on A40: FP32 unmerged/merged max error 3.77e-5 over 10.9M
    # logits. The absolute floor covers FP32 reassociation near zero; see validation/.
    "fp32": {"rtol": 0.0001, "atol": 0.0001},
    "bf16": {"rtol": 0.05, "atol": 0.05},
}

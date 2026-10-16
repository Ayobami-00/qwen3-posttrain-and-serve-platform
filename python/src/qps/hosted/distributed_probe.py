"""Opt-in download-free NCCL/FSDP2/PEFT compatibility proof on two GPUs."""

import copy
import json


def probe():
    from qps.distributed import finite_everywhere, initialize

    initialize()
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    from peft import LoraConfig, get_peft_model
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.tensor import DTensor
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(42)
    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=64,
                _attn_implementation="eager",
            )
        ),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    reference = copy.deepcopy(model) if dist.get_rank() == 0 else None
    accelerator = Accelerator(
        mixed_precision="bf16",
        fsdp_plugin=FullyShardedDataParallelPlugin(
            fsdp_version=2,
            cpu_ram_efficient_loading=False,
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap=["Qwen3DecoderLayer"],
            activation_checkpointing=True,
            mixed_precision_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        ),
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    model, optimizer = accelerator.prepare(model, optimizer)
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    assert actual == {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert all(isinstance(p, DTensor) for p in model.parameters())
    before = {
        n: p.detach().to_local().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    batches = []
    for step in range(2):
        ids = torch.randint(
            0, 128, (2, 16), generator=torch.Generator().manual_seed(100 + step)
        ).to(accelerator.device)
        labels = ids.clone()
        labels[:, :8] = -100
        labels[1, 12:] = -100
        batches.append((ids.cpu(), labels.cpu()))
        rank = dist.get_rank()
        local_labels = labels[rank : rank + 1]
        count = (local_labels[:, 1:] != -100).sum()
        total_count = (labels[:, 1:] != -100).sum()
        loss = (
            model(input_ids=ids[rank : rank + 1], labels=local_labels).loss
            * count
            / total_count
            * 2
        )
        finite_everywhere(loss.detach(), "probe loss")
        accelerator.backward(loss)
        norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
        finite_everywhere(norm, "probe gradient norm")
        optimizer.step()
        optimizer.zero_grad()
    changed = sum(
        (p.detach().to_local() - before[n]).abs().sum().item()
        for n, p in model.named_parameters()
        if n in before
    )
    total = torch.tensor(changed, device=accelerator.device)
    dist.all_reduce(total)
    assert total.item() > 0
    diagnostics = [None] * 2
    dist.all_gather_object(
        diagnostics,
        {
            "rank": dist.get_rank(),
            "device": torch.cuda.current_device(),
            "gpu": torch.cuda.get_device_name(),
            "peakAllocatedBytes": torch.cuda.max_memory_allocated(),
        },
    )
    from qps.distributed import barrier, gather_adapter

    sharded_state = gather_adapter(model)
    parity = None
    if reference is not None:
        reference.to(accelerator.device)
        reference_optimizer = torch.optim.AdamW(
            [p for p in reference.parameters() if p.requires_grad], lr=0.001
        )
        for ids, labels in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reference_loss = reference(
                    input_ids=ids.to(accelerator.device), labels=labels.to(accelerator.device)
                ).loss
            reference_loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            reference_optimizer.step()
            reference_optimizer.zero_grad()
        errors = []
        for name, p in reference.named_parameters():
            if p.requires_grad:
                expected = p.detach().cpu()
                actual = sharded_state[name]
                errors.append((expected - actual).abs().max().item())
                torch.testing.assert_close(actual, expected, atol=0.0002, rtol=0.02)
        parity = {
            "equivalentGlobalBatch": True,
            "maxAdapterAbsoluteError": max(errors),
            "atol": 0.0002,
            "rtol": 0.02,
            "note": "BF16 computation with FP32 reduction; two deterministic masked batches, dropout disabled.",
        }
    barrier()
    result = {
        "passed": True,
        "referenceParity": parity,
        "worldSize": 2,
        "backend": "nccl",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "adapterDeltaL1": total.item(),
        "ranks": diagnostics,
    }
    dist.destroy_process_group()
    return result


if __name__ == "__main__":
    import os

    result = probe()
    if int(os.environ["RANK"]) == 0:
        print(json.dumps(result), flush=True)

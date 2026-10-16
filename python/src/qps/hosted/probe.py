"""Download-free BF16/Qwen3/LoRA compatibility check on the allocated GPU."""

import json


def probe():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This profile requires exactly one BF16-capable CUDA device")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 20 * 1024**3:
        raise RuntimeError("The allocated GPU has insufficient usable memory")
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = get_peft_model(
        Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    ids = torch.randint(0, 128, (1, 16), device="cuda")
    before = {n: p.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad}
    for _ in range(2):
        loss = model(input_ids=ids, labels=ids).loss
        if not torch.isfinite(loss):
            raise RuntimeError("GPU probe produced non-finite loss")
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError("GPU probe produced non-finite gradients")
        optimizer.step()
        optimizer.zero_grad()
    if not any(
        not torch.equal(before[n], p.detach().float())
        for n, p in model.named_parameters()
        if p.requires_grad
    ):
        raise RuntimeError("GPU probe did not update the adapter")
    torch.cuda.synchronize()
    return {
        "passed": True,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "vramBytes": properties.total_memory,
        "computeCapability": [properties.major, properties.minor],
        "loss": float(loss),
        "peakAllocatedBytes": torch.cuda.max_memory_allocated(),
    }


if __name__ == "__main__":
    print(json.dumps(probe()))

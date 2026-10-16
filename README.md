# Qwen3 Post-Training and Serving Platform

An open-source Python application for reproducible Qwen3 LoRA training, evaluation, checkpoint recovery and reference HTTP serving.

**v0.1: local CPU and NVIDIA A40 GPU workflows validated.** The CPU proof uses a tiny randomly initialized Qwen3 model and real optimizer updates. The GPU validation completed the Qwen3-0.6B/GSM8K recipe, recovered the optimizer from checkpoint 25, and verified export, HTTP parity, streaming and server restart. See the [GPU validation evidence](validation/gpu-2026-10-15.json).

On 128 held-out examples from GSM8K's training split, strict answer accuracy changed from 46/128 (35.94%) to 72/128 (56.25%); parse failures changed from 63 to 4. Answer-format adherence contributes substantially to that gain. This is a small systems-validation sample, not an official test benchmark. All 33 CPU checks passed on both macOS and Linux.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Run commands from the repository root on macOS or Linux:

```bash
uv sync --project python --python 3.12 --frozen --extra train --extra serve --group dev
uv run --project python --no-sync qps --help
```

The checked-in lock selects the dependencies. `--no-sync` in subsequent commands preserves the installed training/serving extras. For CLI/configuration-only use, omit the extras when installing; `qps validate` does not import PyTorch or download weights.

## Complete local proof

```bash
uv run --project python --no-sync qps smoke --work-dir .qps/smoke
```

This generates an offline tiny Qwen3 snapshot and arithmetic data, evaluates the parent, trains a LoRA adapter for four optimizer steps, saves full resumable checkpoints, verifies a merged export, evaluates it, serves it over real loopback HTTP, checks streaming and offline/HTTP parity, restarts the server and shuts it down. Each invocation gets a new run name. No model/dataset downloads are required after dependency installation.

The command prints its run/artifact IDs and saves a `smoke-*-report.json` in the work directory. The report explicitly records `gpu_validated: false`. Production-model performance and quality need the later GPU acceptance run.

## Run the steps individually

```bash
uv run --project python --no-sync qps fixtures create --directory .qps/fixture --name tiny-cpu
uv run --project python --no-sync qps validate --spec examples/local/tiny-cpu.yaml
uv run --project python --no-sync qps run --spec examples/local/tiny-cpu.yaml --wait
uv run --project python --no-sync qps runs show tiny-cpu
uv run --project python --no-sync qps models list --run tiny-cpu
uv run --project python --no-sync qps deploy create --from-run tiny-cpu --name tiny-cpu --port 8000
uv run --project python --no-sync qps evaluate --deployment tiny-cpu
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-cpu","messages":[{"role":"user","content":"What is 2 plus 3?"}],"max_tokens":8,"stream":true}'
uv run --project python --no-sync qps deploy stop tiny-cpu
```

Run names are unique and history is never overwritten. Choose a new name in the YAML for a new experiment. Deployment names may be restarted after stopping; they identify a verified merged artifact. Global `--state-dir` belongs before the command, for example `qps --state-dir .qps/smoke runs list`. Local source and artifact paths resolve against the working directory, not the YAML file's directory.

## Local logging and recovery

Training emits a readable terminal event at **every optimizer step**, including loss, learning rate, gradient norm, processed tokens and elapsed time. CUDA peak allocated memory is recorded on CUDA; it is `null` on CPU. These losses are training losses, not answer accuracy, and need not decrease monotonically.

```bash
# Background execution, then follow structured progress from another terminal.
uv run --project python --no-sync qps run --spec examples/local/tiny-cpu.yaml --no-wait
uv run --project python --no-sync qps runs logs tiny-cpu --follow
uv run --project python --no-sync qps runs cancel tiny-cpu
uv run --project python --no-sync qps runs resume tiny-cpu --checkpoint latest
uv run --project python --no-sync qps runs retry tiny-cpu --phase evaluation
```

The background example assumes a fresh run name; it is an alternative to the earlier foreground run. Cancellation may discard work since the last complete checkpoint. Resume accepts a registered checkpoint ID or its verified directory, and requires the same specification and environment. `runs resume` without a checkpoint continues verified completed phases and uses the latest checkpoint if training was interrupted. Evaluation-only retry reuses model outputs without training again.

Runtime files are ignored by Git:

```text
.qps/
├── metadata.sqlite                     # Authoritative run/attempt/event/artifact metadata
├── runs/<run-id>/
│   ├── submitted.yaml, spec.json, resolved.json
│   ├── events.jsonl, metrics.jsonl      # Ordered projections rebuilt from SQLite
│   ├── comparison.json
│   └── attempts/<n>/*.log              # Worker stdout, stderr and exceptions
├── datasets/<artifact-id>/              # Prepared split files and immutable manifests
├── artifacts/<artifact-id>/             # Parent, adapter, checkpoint, merged model and reports
└── deployments/                         # Serving logs and HTTP evaluation reports
```

Custom work/artifact roots are recorded in the resolved specification. Checkpoints preserve optimizer, scheduler, RNG and trainer progress, and are distinct from serving exports. Artifacts are staged, checksummed and finalized before registration. A failed evaluation retains the trained artifacts. The total active coordinator wall-time budget includes all phases and persists across retries; downtime between attempts is excluded. An exhausted run needs a new submission.

Merged exports contain FP32 master weights. Export verification compares the unmerged FP32 adapter model with the reloaded FP32 export; training and generation retain their configured precision. The artifact records these dtypes separately. The [GPU merge precision investigation](validation/merge-precision-2026-10-15.json) explains the checked-in FP32 tolerance. Offline/HTTP generation parity compares the same merged artifact at the same execution precision.

## Reference serving contract

The server verifies and loads one merged model at startup. It exposes `/healthz`, `/readyz`, `/v1/models`, `/v1/completions` and `/v1/chat/completions`. Requests support one text prompt or text chat, `max_tokens` (1–512), greedy `temperature: 0`, `n: 1`, and optional SSE streaming. Other parameters are rejected. Context overflow returns 422, unknown models 404 and concurrent generation 429. Streaming ends with a finish-reason chunk and `[DONE]`; disconnects cancel generation.

Serving binds to `127.0.0.1` only. Use an SSH tunnel on a remote execution host. There is one active generation per process. The artifact's tokenizer/template and non-thinking profile apply to chat rendering; completions accept a raw prompt. SQLite deployment records bind endpoints to exact artifact IDs. A fresh readiness probe is required on every restart.

## NVIDIA GPU validation

On a Linux NVIDIA host, install Python 3.12, a driver compatible with the CUDA runtime selected by the locked Linux PyTorch package, and the same locked extras. Check compatibility before downloading the model:

```bash
uv run --project python --no-sync python -c \
  'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
uv run --project python --no-sync qps validate --spec examples/local/gsm8k-lora.yaml
uv run --project python --no-sync qps run --spec examples/local/gsm8k-lora.yaml --wait
```

On an already provisioned compatible GPU host, the opt-in acceptance test runs the real recipe and verifies its HTTP deployment:

```bash
QPS_RUN_GPU=1 uv run --project python --no-sync pytest python/tests -m gpu -q -s
```

The recipe requests one BF16 GPU, LoRA rank 16, 100 optimizer steps, sequence length 1,024, microbatch 1, accumulation 16, checkpoints every 25 steps and 1,024/128 training/validation examples drawn from GSM8K's training split. Hub revisions are pinned before execution. The official test split is untouched. Truncated/malformed generations count as failures, with their own counters. Public benchmark pretraining contamination remains possible.

The one-off Runpod validation used an available A40 at $0.59/hour and an observed account balance decrease of about $0.47. Peak allocated CUDA memory during training was 1.83 GiB; this excludes allocator reservations and memory needed in other phases. This validates the recorded A40/software configuration. Other GPU configurations still require their own compatibility checks.

The complete results were copied locally and all 20 registered artifacts were verified before the Pod was terminated. The local evidence directory is `.qps/runpod-validation/`, with the complete compressed archive, readable reports and per-step metrics. Allow disk for the Python environment, source caches, durable parent snapshot, merged FP32 weights and checkpoints. The application does not provision Runpod resources; the validation used a separate one-off controller. Preserve results before terminating rented compute. Registry paths are execution-host absolute paths; archived artifact files are self-contained, while relocating an entire registry is not automated.

## Development checks

```bash
uv run --project python --no-sync ruff check python/src python/tests
uv run --project python --no-sync ruff format --check python/src python/tests
uv run --project python --no-sync pytest python/tests -m 'not gpu' -q
uv run --project python --no-sync qps schema --output contracts/run.schema.json
```

CPU tests run without external model/dataset access or paid services. CI covers macOS and Linux; a successful local run does not by itself establish Linux/CUDA compatibility. TensorBoard/W&B, dashboard/authentication, provider orchestration, distributed training, additional model families and Rust inference are later work.

## License and project status

The code is licensed under [Apache 2.0](LICENSE). Downloaded models, datasets and dependencies retain their own licenses and attribution requirements.

This is an independent community project and is not affiliated with or endorsed by the Qwen team.

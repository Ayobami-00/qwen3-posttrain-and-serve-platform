# Python application

The `qps` package contains the local training-to-serving implementation and the
hosted API/worker bridge. Both execution paths share the same phase functions.
Run the following commands from the repository root.

## Install and verify locally

```bash
uv sync --project python --python 3.12 --frozen --extra train --extra serve --extra hosted --group dev
uv run --project python --no-sync qps validate --spec examples/local/tiny-cpu.yaml
uv run --project python --no-sync qps smoke --work-dir .qps/smoke
```

The smoke command creates a tiny random Qwen3 snapshot and arithmetic fixtures,
performs real adapter updates, verifies an export and tests HTTP serving. Once
dependencies are installed, it needs no model downloads or paid resources.
It establishes functional CPU behavior, not pretrained model quality or CUDA
compatibility. `--no-sync` preserves the extras selected during installation.

## Follow one training execution

| Module | Responsibility |
| --- | --- |
| [specs.py](src/qps/specs.py) | Validate versioned recipes and execution combinations |
| [cli.py](src/qps/cli.py) | Local commands plus the `remote` and `platform` command groups |
| [pipeline.py](src/qps/pipeline.py) | Coordinate attempts and isolated phase processes |
| [worker.py](src/qps/worker.py) | Run one ordinary phase and write its output |
| [stages.py](src/qps/stages.py) | Prepare, baseline, export and candidate evaluation |
| [training.py](src/qps/training.py) | TRL/PEFT training, step events, checkpoints and adapter publication |
| [distributed_worker.py](src/qps/distributed_worker.py) | One torchrun rank; rank zero owns the journal |
| [distributed.py](src/qps/distributed.py) | FSDP2 configuration, collectives and checkpoint layout checks |
| [store.py](src/qps/store.py) | SQLite execution journal and JSONL projections |
| [artifacts.py](src/qps/artifacts.py) | Checksummed local artifact publication and verification |
| [server.py](src/qps/server.py) | Bounded, single-model HTTP generation |

The phases are `prepare`, `baseline`, `training`, `export`, `evaluation`.
Each ordinary phase runs in a separate process so model memory is released
between phases. FSDP2 changes the training phase to two torchrun ranks on one
host; baseline/candidate generation and export remain separate ordinary phases.

The coordinator reads each worker's completed output, verifies its artifact
references, and only then commits phase success. Worker exit status alone does
not establish that a phase produced usable outputs.

## Supervision, progress and recovery

[data.py](src/qps/data.py) groups duplicate questions before deterministic split
selection. Prompt labels and padding labels use `-100`; the complete assistant
solution and end token are supervised. Overlong examples are rejected rather
than losing their final answers through truncation.

`maxSteps` counts optimizer updates. Effective batch size is microbatch times
accumulation times rank count: `1 × 16 × 1` and `1 × 8 × 2` both give 16.

Per-step events include loss, reported learning rate/gradient norm, token counts
and timing. SQLite is authoritative locally; `events.jsonl` and `metrics.jsonl`
are projections. A checkpoint includes optimizer, scheduler, RNG and progress
state, plus exact parent/dataset/recipe/software identity. Resume validates those
identities and retains global-step continuity. FSDP2 recovery requires two ranks.

## Serving and hosted use

Exports retain a normal PEFT adapter and a merged FP32 Safetensors master.
The export phase reloads the merged model and checks fixed-input logits.
[runtime.py](src/qps/runtime.py) verifies a standalone bundle without requiring
the original training registry. GPU serving uses the configured BF16 execution
profile and one active generation, even for a model trained with FSDP2.

See the [hosted module guide](src/qps/hosted/README.md) for remote publication,
leases and independent model ownership, and [end-to-end checks](../tests/e2e/README.md)
for the boundaries between CPU, Kubernetes and GPU acceptance.

## Focused regression checks

The [Python test guide](tests/README.md) separates quick parsing/preparation,
artifact-integrity and journal checks from real CPU training and service tests.
It also identifies which fixtures load no ML libraries and which run the smoke
workflow. Use the recorded release evidence for GPU acceptance; passing a small
byte-payload or tokenizer-double test does not establish model compatibility.

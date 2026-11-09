# Shared contracts

Contracts describe the boundary between the dashboard, Python control plane,
Go operator, remote workers and future Rust inference engine. They carry recipes
and identities, never cloud keys or signed object URLs in research settings.

## Hosted workflow

| Contract | Main use |
| --- | --- |
| [Run](run.schema.json) | Model, dataset, training, evaluation and local execution recipe |
| [Hosted submission](hosted-run.schema.json) | Recipe plus compute profile, dataset version and idempotency key |
| [Worker events](worker-events.schema.json) | Ordered attempt events and durable replay |
| [Model version](model-version.schema.json) | Independently deployable published model |
| [Deployment](deployment.schema.json) | Model/artifact selection and serving limits |
| [Deletion operation](deletion-operation.schema.json) | Durable asynchronous cleanup progress |
| [Python backend](backend.schema.json) | Existing Python serving capability contract |
| [OpenAPI](openapi.json) | Versioned HTTP interfaces used to generate frontend types |

The Python definitions live in [specs.py](../python/src/qps/specs.py) and
[contracts.py](../python/src/qps/contracts.py). Regenerate their public schemas
from the repository root with:

```bash
uv run --project python --no-sync python tools/generate_contracts.py
npm --prefix web run contracts
```

Inspect the resulting changes together. These commands do not implement missing
backend behaviors or regenerate the separately reviewed Rust contract pack.

## Rust inference contract pack

| Document | Example | Purpose |
| --- | --- | --- |
| [Runtime schema](inference-runtime.schema.json) | [Runtime fixture](fixtures/inference/runtime.json) | Deployment request, local artifact directory and execution intent |
| [Backend schema](inference-backend.schema.json) | [Backend fixture](fixtures/inference/rust-backend.json) | Engine-owned supported profiles, file formats and limits |
| [Model-profile schema](model-profile.schema.json) | [Qwen3 profile](model-profiles/qwen3-0.6b-v1.json) | Required subset of exported model configuration |

These documents use required fields and reject additional properties. QPS names
use camelCase; the embedded Hugging Face model configuration keeps its original
field names. The runtime example uses synthetic IDs and an all-zero hash; it is
not a real artifact location.

Keep three checks distinct:

1. **Document validity:** supported field types, values, identifier formats and
   list/limit rules. Rust currently implements runtime/backend document checks.
2. **Compatibility:** requested backend, model profile and complete execution
   tuple must match advertised capabilities. This is the next Rust step.
3. **Artifact verification:** original manifest bytes, inventory, file hashes,
   tokenizer/prompt identity and model/tensor compatibility must agree.

Schema validity alone permits values the initial engine does not advertise.
TP=2 and CPU are representable, but the initial Rust target is CUDA/BF16/TP=1.
Stored FP32 weights and BF16 execution describe different stages; loading must
not modify the immutable master export.

The new documents preserve the existing version-1 artifact envelope. Keep its
manifest hash over original file bytes; keep the template fingerprint as SHA-256
over the project's canonical JSON encoding of the template string. Model profile
`requiredConfig` is a subset match, not a ban on ordinary extra model metadata.

Do not replace the existing Python backend schema with the future Rust descriptor:
their compatibility and rollout are separate. See the [Rust implementation guide](../rust/README.md).

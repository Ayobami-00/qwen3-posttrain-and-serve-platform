# End-to-end acceptance

Different gates establish different facts. CPU fixture success is not evidence
of CUDA, NCCL, real-model quality or provider cleanup. Recorded release reports
retain the tested source/image identities and any incomplete gates.

| Gate | Main driver | What it exercises |
| --- | --- | --- |
| Local smoke | `qps smoke` | Offline tiny Qwen3, real updates, checkpoints, export and loopback serving |
| Hosted CPU | [hosted_cpu.py](hosted_cpu.py) | API, S3 emulator, disposable workers, loss/recovery and model/deletion lifecycles |
| Local browser | [browser_server.py](browser_server.py) and [workflow.spec.ts](../../web/tests/workflow.spec.ts) | Owner/public boundaries and the dashboard workflow with a fake coordinator |
| Linux Kubernetes | [kubernetes_acceptance.py](../../tools/kubernetes_acceptance.py) | Real Go controller, Kind, PostgreSQL, S3 emulator and browser workflow |
| Live single GPU | [hosted_live.py](hosted_live.py) | Real hosted/provider training, checkpoint recovery and serving |
| Live distributed | [v04_live.py](v04_live.py) | Same-host two-GPU experiments and distributed recovery acceptance |

## Local CPU checks

Run from the repository root after installing the locked Python train/serve/hosted
extras and development group:

```bash
uv sync --project python --python 3.12 --frozen --extra train --extra serve --extra hosted --group dev
uv run --project python --no-sync pytest python/tests -m 'not gpu' -q
uv run --project python --no-sync qps smoke --work-dir .qps/smoke
uv run --project python --no-sync python tests/e2e/hosted_cpu.py
```

These use generated model/data fixtures and do not provision paid cloud compute.
The hosted CPU proof uses an HTTP S3 emulator and a SQLite metadata adapter;
PostgreSQL migrations/concurrency are checked separately by supplying
`QPS_TEST_POSTGRES_URL` to the PostgreSQL tests. It does not validate Kubernetes.

## Browser fixture

Start the fixture API in one terminal and Next.js in another:

```bash
uv run --project python --no-sync python tests/e2e/browser_server.py
```

```bash
npm --prefix web ci
npm --prefix web run dev
```

Then install the test browser if needed and run the workflow:

```bash
npm --prefix web exec playwright install chromium
npm --prefix web run test:e2e
```

The [Playwright configuration](../../web/playwright.config.ts) defaults to the
loopback dashboard. The fixture uses development authentication, not Cognito.
Real Cognito and public HTTPS streaming are separate live acceptance checks.

## Linux and provider boundaries

The managed [build](../../containers/buildspec.yml) runs Linux acceptance before
building/deploying the GPU runtime. Kind/controller tests use disposable local
resources in that build environment; fake-provider tests do not establish real
GPU capacity or pricing.

Live drivers are paid acceptance tools, not ordinary test-suite commands. They
require an existing installation, owner credentials, agreed cumulative budgets,
provider access and cleanup scheduling. Distributed live acceptance additionally
requires explicit `QPS_LIVE_FSDP2=1` opt-in. Keep credentials and signed URLs out of
screenshots/reports and verify cleanup independently of a successful training result.

The [v0.4 recorded report](../../validation/v04-hosted-2026-10-16.json) passed CPU,
Kubernetes/browser and hosted lifecycle checks, but its two-GPU compatibility probe
timed out before training. A successful rerun with exact image/hardware identities
is still required for FSDP2, distributed optimizer recovery and the four-GPU proof.
The earlier [v0.3 report](../../validation/v03-hosted-2026-10-16.json) records the
separate single-GPU hosted acceptance.

Small local evidence and browser screenshots belong under ignored `.qps/`.
Release summaries belong in [validation](../../validation); large weights stay
on the worker or in private S3. A historical URL is not a live health check.

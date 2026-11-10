# QPS Go controller

The leader-elected controller reconciles namespaced `TrainingRun`, `ComputeAllocation` and `ModelDeployment` CRDs. It writes lifecycle observations to Kubernetes before projecting status into PostgreSQL. The API's transactional outbox makes dispatch restartable.

PostgreSQL transactions own FIFO admission and cost reservations. Create intent and exact ownership are persisted before a non-idempotent Runpod request. An uncertain response is reconciled by the allocation's committed name; it never causes a blind second create. Cancellation/deadlines trigger deletion, and reservations remain until absence is confirmed. An independent Lambda uses the durable S3 allocation ledger after EKS loss.

Profiles use one GPU for ordinary training/serving, two GPUs on one host for FSDP2 training, or a disposable Kubernetes CPU Job for acceptance. GPU workers do not join EKS. Only the controller/janitor hold the Runpod management key. Workers receive fenced credentials and approved object access; private ECR credentials are short lived and removed during cleanup.

## Reconcile one workload

| Component | Responsibility |
| --- | --- |
| [Manager](cmd/operator/main.go) | Namespace-scoped cache, leader election, health endpoints and reconciler registration |
| [Workload reconciler](internal/controller/reconcilers.go) | Observe training/deployment intent, obtain admission, create child allocation and project status |
| Allocation reconciler in the same module | Record provider creation intent, observe the exact resource and complete external cleanup |
| [Runpod adapter](internal/provider/runpod.go) | Catalog quotes, bounded topology, provider requests and registry credentials |
| [CPU adapter](internal/controller/cpu.go) | Kubernetes Jobs for download-free fixture execution |
| [Control-plane client](internal/controlplane/client.go) | Authenticated API calls rather than direct database writes |

The API dispatches an owner ID, execution generation and action through the
database outbox. The workload reconciler reads the full current intent from the
API. A repeated reconciliation observes existing allocation state before
creating anything new. Status is written to Kubernetes before its database
projection; the database remains authoritative for submitted intent/admission.

## Admission and replacement

Admission counts GPU units, not Pod count. With the current default allowance of
four GPUs, two FSDP2 experiments occupy `2 + 2` units; one FSDP2 experiment can
coexist with two single-GPU allocations. Training and serving share the budget.
CPU fixtures have a separate bounded concurrency limit.

The quote must match the requested topology and fit the per-GPU price ceiling.
The API transaction locks the project budget, enforces FIFO admission, reserves
the full rate/storage/cleanup allowance and records the allocation before
provisioning. Unconfirmed termination remains counted. GPU startup count is also
bounded because each cold worker image pull contributes to the AWS forecast.

FSDP2 requests exactly two devices in one provider Pod. The controller does not
substitute separate single-GPU hosts. One eligible infrastructure replacement can
resume a remotely committed checkpoint after prior cleanup is confirmed. It
retains the run's remaining runtime and creates a new fenced generation.
Torchrun launcher restarts are disabled so the platform owns replacement policy.

## Cleanup ownership

The `qps.dev/cleanup` finalizer prevents a Kubernetes deletion from completing
before external allocations have stopped. Creation outcome, provider name/ID,
image identity, returned rate and GPU count are reconciled explicitly. An
ambiguous create response does not justify a second create request.

The independent [Lambda janitor](../infra/lambda/janitor.py) uses durable owned
allocation records after EKS loss. Training result and cleanup confirmation are
separate states. Product deletion retains spending history through the accounting
ledger; it must not turn a previously billed run into fresh budget.

## Checks and acceptance

```bash
GOTOOLCHAIN=go1.27.2 go -C operator test ./...
GOTOOLCHAIN=go1.27.2 go -C operator build ./cmd/operator
```

Tests cover quotes, ambiguous accepted creation, restart, duplicate prevention and ownership-safe deletion using fake Kubernetes/provider interfaces. Linux acceptance runs the actual controller against Kind/PostgreSQL/S3. CRDs and namespace-scoped RBAC are in `deploy/helm/qps`.

Fake-provider/controller tests do not establish CUDA or NCCL compatibility.
The [recorded v0.4 evidence](../validation/v04-hosted-2026-10-16.json) reports a
two-GPU compatibility-probe timeout before training; real FSDP2 acceptance is
still pending. See the [hosted application guide](../python/src/qps/hosted/README.md)
for leases, worker events and object publication.

Runpod REST v2 has no atomic maximum-price or automatic-expiry field. Admission, returned-rate verification, worker leases, runtime accounting and independent cleanup enforce the application's bounds. Provider billing evidence remains separate from estimates.

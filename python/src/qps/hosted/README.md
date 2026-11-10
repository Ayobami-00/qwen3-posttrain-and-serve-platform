# Hosted control plane and worker bridge

The hosted path wraps the local training and serving implementations. EKS hosts
the API/dashboard/controller; GPU workers remain external Runpod allocations.
CPU fixtures instead execute as bounded Kubernetes Jobs.

## Ownership and durable state

| System | Authority |
| --- | --- |
| PostgreSQL | Submitted intent, attempts, ordered events, registry, outbox and accounting |
| Kubernetes | Reconciled live workload lifecycle and cleanup finalizers |
| S3 | Verified artifact bytes, version IDs and recovery/cleanup ledgers |
| Worker SQLite | Disposable execution journal reconstructed from hosted state |
| Forge | Projection of platform events and per-attempt tracking |

Run, attempt, allocation, artifact, model version and deployment are distinct
identities. A replacement attempt fences its predecessor. Published model
versions retain serving configuration and references independently of the source
experiment's product rows.

## Follow a submission

1. [api.py](api.py) authenticates the owner, validates `HostedSubmission`, pins
   model/dataset revisions and selects an immutable worker image.
2. [database.py](database.py) commits the run, initial attempt and dispatch outbox
   in one transaction. Replayed idempotency keys must have identical contents.
3. [kubernetes.py](kubernetes.py) claims outbox rows and applies `TrainingRun` or
   `ModelDeployment` resources. Expiring claims permit restartable delivery.
4. The [Go controller](../../../../operator/README.md) obtains admission and
   reconciles a `ComputeAllocation` with the provider.
5. [agent.py](agent.py) bootstraps approved artifacts and outputs, reconstructs a
   local journal and launches the existing pipeline. It forwards events and
   phase outputs after publication has made referenced artifacts usable.

The preparation identity includes resolved input revisions, sequence/generation
policy and worker image. Compatible experiments can reuse committed prepared
inputs while their research settings and outputs remain independent.

## Worker trust and publication

[auth.py](auth.py) implements Cognito PKCE owner sessions and CSRF checks. The
controller uses an internal credential; workers use allocation-specific tokens
and lease generations. Workers do not receive AWS administrator credentials or
the Runpod management key. The agent strips hosted secrets from training-phase
subprocess environments.

Heartbeats renew a bounded lease and carry telemetry. Both the worker watchdog
and control plane enforce expiry. Stale generations cannot ingest events or
commit artifacts. Events are replayable by sequence; a reused sequence with
different contents is rejected.

[objects.py](objects.py) issues scoped staging upload URLs, reads back and verifies
file size/SHA-256, copies final object versions, writes the manifest and commits
the registry entry after rechecking fencing. Incomplete uploads are not valid
resume/deployment inputs. PostgreSQL and S3 are not one atomic transaction;
ordering, idempotency, verification and reconciliation handle interrupted work.

## Model, deployment and deletion lifecycles

[lifecycle.py](lifecycle.py) promotes evaluated exports into independent models
and tracks artifact references. [deletion.py](deletion.py) reconciles durable
cleanup operations: fence work, confirm compute cleanup, remove owned Kubernetes
resources, clean tracking, wait for upload grants, purge unreferenced S3 versions,
then delete product metadata. Accounting and deletion tombstones survive.

Experiment deletion preserves published models/deployments. Model deletion
cascades to its deployments. Stopped-deployment deletion preserves its model.
Historical backups and infrastructure logs keep their documented expiry.

## Inference and monitoring

The deployment agent starts the Python server on loopback port 8001, verifies
readiness identity, then starts the authenticated [relay](relay.py) on port 8000.
The owner gateway in `api.py` resolves the deployment and preserves streaming and
disconnect propagation. [http.py](http.py) coordinates upstream disconnects.

[forge.py](forge.py) mirrors one tracking run per attempt and groups by experiment;
weights remain private in S3. Native charts read platform events independently.
The agent collects NVML telemetry and rank-health files; the API exposes labeled
Prometheus series including explicit freshness/availability. External workers do
not need to join the cluster to appear on the GPU dashboard.

See [installation](../../../../infra/README.md) for role/secret/build setup and
[acceptance checks](../../../../tests/e2e/README.md) for what each test gate proves.

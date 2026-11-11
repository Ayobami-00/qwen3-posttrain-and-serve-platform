# Hosted installation and lifecycle

The CPU platform runs on EKS in London. GPU containers remain external Runpod resources. PostgreSQL owns submissions, attempts, events, registry metadata and dispatch intent. Kubernetes owns live workload lifecycle. Verified versioned S3 manifests determine whether an artifact is usable. Worker SQLite journals are disposable.

## Setup

Use Python 3.12, uv 0.11.31 and AWS CLI v2 with an authenticated profile. The installer downloads checksum-verified Terraform 1.16.6. Set the account guard and credential file paths in `examples/hosted/platform.example.yaml`:

```bash
uv sync --project python --python 3.12 --frozen --extra hosted --group dev
chmod 600 ~/.config/qps/runpod-api-key ~/.config/qps/wandb-api-key
uv run --project python --no-sync qps platform doctor --config examples/hosted/platform.example.yaml
uv run --project python --no-sync qps platform init --config examples/hosted/platform.example.yaml
uv run --project python --no-sync qps platform plan
uv run --project python --no-sync qps platform apply
uv run --project python --no-sync qps platform status
```

Commit the tested source first: cloud builds use an immutable Git archive. Never put key strings in YAML, Terraform variables, images or Git. AWS CLI login refresh is bridged into Terraform; hosted services and managed jobs use project roles. No permanent root keys or temporary automation user is created. Secret containers are Terraform resources; values are injected outside Terraform state.

`init` creates private state/artifact/report/source/log buckets, ECR repositories, secret containers, build/deployment roles and independent cleanup schedules. It migrates bootstrap state to versioned S3 with native lockfiles. Generated owner credentials are stored in ignored `.qps/platform/test/owner-credentials.json` with mode 0600. Signup is disabled and no email is sent.

`apply` requires passing local acceptance evidence. Linux CodeBuild builds CPU images and runs disposable Kind/PostgreSQL/S3/browser acceptance before building the CUDA image. Pretrained weights and CUDA images stay off the Mac. The installer recalculates costs using actual image size before creating EKS. The foundation creates two private subnets, one NAT gateway, EKS 1.36 with two fixed m7i.large nodes, private PostgreSQL 18.6 on db.t4g.micro, and CloudFront HTTPS with an internal ALB. Private cluster operations run in VPC-attached service-role jobs.

Deployment-only changes can reuse a successful recorded image build when its runtime inputs are identical. The installer verifies the build's CPU gate and source identity; it retains the original image digests and build commit separately from the infrastructure source commit. Runtime source or dependency changes require a new build. Application Pods explicitly set both AWS region environment variables so Python and Go use the same region with Pod Identity credentials.

The environment has a 24-hour maximum lifetime and an earlier budget cutoff when needed to keep the forecast below $20. The $30 AWS allowance includes builds, transfers, retention and cleanup. The forecast includes configured cold GPU image pulls, historical transfers and no free-tier credits; use the recorded build's actual image size rather than a fixed pull-count assumption. Billing observations are delayed; these safeguards are not a guaranteed hard financial cap.

For a bounded validation retry, `validationGpuImagePulls` can reserve additional cold image transfers. The installer shortens the existing expiry to cover them; it preserves the original AWS and Runpod spending limits. The live serving proof reserves 30 minutes across startup and restart, within the 45-minute allocation ceiling.

## Use

Open the returned dashboard URL and use the protected owner credential file. Anonymous visitors see approved summaries, settings and charts. Owner login is required for controls, raw events, downloads, Grafana and inference.

The curated recipe remains Qwen3-0.6B/GSM8K SFT with query/value LoRA targets. Clone experiments to change supported settings. Preparation identity includes pinned Hub revisions, tokenizer/template, split and sequence policy, and worker image; compatible concurrent runs share immutable data and parent artifacts.

The v0.4 Go controller admits work FIFO into four shared GPU units across training and serving. Single-GPU work consumes one unit; a two-GPU FSDP2 Pod consumes two. CPU fixtures have a separate bounded concurrency limit. It reserves projected cost before provisioning and keeps uncertain deletions counted. Current application defaults cap cumulative Runpod spend at $10, price at $1/GPU-hour and runtime at 45 minutes per allocation. Recovery retains the original run's remaining runtime and previously recorded spending. A compatibility probe runs before normal training; one eligible infrastructure replacement may restore a committed checkpoint. These are code/configuration limits, not a statement of current provider balance or availability.

Native charts show durable per-step metrics and GPU telemetry. Forge mirrors one run per attempt, grouped by platform run; weights remain private in S3. Select a verified evaluated merged artifact to deploy. A loopback Python model process sits behind an authenticated Runpod relay and owner gateway. The supported OpenAI subset preserves SSE, busy responses and cancellation, with at most 2,048 context and 512 output tokens. Stop GPU deployments immediately after testing.

Prometheus scrapes the API and Go controller. Alertmanager exposes service, reconciliation and stale-worker alerts through Grafana; the initial receiver sends no external notifications. Local Helm chart contents are fingerprinted in Terraform so template-only changes apply reproducibly.

```bash
qps remote login --url https://YOUR_DISTRIBUTION.cloudfront.net
qps remote run --spec examples/hosted/gsm8k-lora.yaml
qps remote --help
```

## Backup and cleanup

```bash
qps platform backup
qps platform destroy
# Only an empty registry can be restored:
qps platform restore
```

Destruction invokes independent GPU cleanup, backs up metadata, removes Kubernetes releases and deletes the owned foundation. The Lambda janitor and allocation ledger remain independent of EKS and local browser authentication. They retry unconfirmed provider cleanup after CPU teardown. Training success and cleanup confirmation are recorded separately. Deletion requires committed project ownership; unrelated resources are preserved.

Managed PostgreSQL backups use a repeatable-read snapshot across all metadata tables, including when submissions or metrics arrive during export. Backups exclude browser sessions and are read verified after private S3 publication.

Terraform owns the three QPS CRDs and upgrades their schemas before the application release. Existing QPS definitions are adopted through declarative import blocks; fresh clusters create them directly. Recovery actions include `resume` and `retry-evaluation`, matching the API dispatch contract.

Bootstrap state, private evidence buckets, ECR and idle managed roles/jobs remain for recovery. Project secret containers are removed and recurring cleanup is disabled after compute absence is confirmed; an unresolved provider cleanup retains the janitor credentials. Conservative retention costs are included in the forecast. Models/checkpoints and images expire after seven days; metadata backups/reports after 30 days. Obsolete versions and unfinished uploads have bounded lifecycle rules. `destroy` deliberately does not empty evidence buckets. Preserve remote state and retained evidence before any later full bootstrap removal.

## Verification

Run the Python CPU suite, `python tests/e2e/hosted_cpu.py`, Go tests and dashboard checks. The HTTP CPU proof includes real training, worker loss, optimizer recovery, S3 publication, export parity, serving, streaming, disconnect cleanup, restart and backup/restore. It explicitly does not validate Kubernetes or GPUs. `QPS_TEST_POSTGRES_URL` enables the real migration/concurrency test; CI supplies PostgreSQL 18.6. Linux Kubernetes/browser acceptance runs before EKS creation. Live acceptance separately records overlapping GPU experiments, checkpoint replacement, Forge lineage, serving parity, measured costs and cleanup.

Reusable modules live in `terraform/core/modules/aws`; bootstrap and foundation/addons/platform roots have separate S3 state keys. Provider locks, chart versions, EKS add-on versions and container base manifests are pinned. `/docs/` remains ignored.

The provisioner includes the EC2 VPC security-group lookup required by ALB creation. EKS launch-template validation receives `RunInstances` only for the project's exact template, with instances restricted to `m7i.large`. A pinned Terraform time resource waits 30 seconds after changes to that policy before creating nodes, avoiding IAM propagation failures on a fresh install.

Forge team privacy is preserved. A private-only team receives private tracking and an owner report link; anonymous visitors continue to use native platform charts. Public report embedding is enabled only when the team's policy permits public projects.

The disposable CodeBuild Kind cluster explicitly permits its host's cgroup v1 configuration. EKS uses AL2023 nodes with cgroup v2. To run the browser proof against the live installation, set `QPS_BROWSER_URL` and `QPS_LIVE_OWNER_FILE` to the returned HTTPS URL and protected owner credential file, then run `npm --prefix web run test:e2e`. The live GPU driver is `python tests/e2e/hosted_live.py`; it persists request identities, interrupts an exactly verified owned worker at checkpoint 25, downloads only small JSON evidence, and stops its allocations in cleanup.

The [v0.3 hosted acceptance](../validation/v03-hosted-2026-10-16.json), recorded on 2026-10-16, passed on EKS and external A40 workers. Its first serving restart produced no Pod; that allocation was stopped and reconciled before one bounded replacement proof passed. Incremental HTTP evidence survives a later acceptance failure. Provider billing is delayed, so the report distinguishes observed billing from the platform estimate and the AWS lifetime forecast. A repeat Terraform apply reported zero resource changes. GPU absence was independently confirmed by the Lambda janitor; the report records the CPU teardown deadline and cleanup state at that time.

The separate [v0.4 evidence](../validation/v04-hosted-2026-10-16.json) records
successful platform/CPU lifecycle checks and an incomplete distributed GPU gate.
Its two-A40 probe timed out before training; real FSDP2 execution, distributed
recovery and that release's trained Qwen3 serving proof remain unverified.
Recorded URLs, deadlines and cleanup observations describe a historical temporary
environment and do not establish current resource availability or completed AWS
teardown. Do not extend a recorded billing window to repeat acceptance.

## Infrastructure layers

| Terraform root | Responsibility |
| --- | --- |
| [Bootstrap](terraform/bootstrap/main.tf) | State/storage buckets, ECR, roles, secret containers, builds and independent cleanup schedules |
| [Foundation](terraform/deployments/test/foundation/main.tf) | Network, private cluster/database access, EKS, RDS and CloudFront/internal ALB |
| [Addons](terraform/deployments/test/addons/main.tf) | Load-balancer controller, Prometheus/Grafana, Loki and log collector |
| [Platform](terraform/deployments/test/platform/main.tf) | CRD upgrades and the application Helm release |

Reusable [AWS modules](terraform/core/modules/aws) implement shared infrastructure.
The [Helm chart](../deploy/helm/qps) runs API, web and operator Deployments with
distinct service identities. The API migration init container upgrades PostgreSQL
before starting the application. Terraform owns ALB target groups; target bindings
connect Kubernetes service endpoints to them.

The [managed build](../tools/cloud_build.py) resolves image digests from a committed
source archive and records the Linux CPU/Kubernetes gate. Runtime input changes
require a new build; infrastructure-only reuse preserves exact image identity.
The [operations runner](../python/src/qps/hosted/operations.py) performs private
cluster operations with service-role credentials. The
[janitor](lambda/janitor.py) independently reconciles owned provider allocations
from S3 records. It does not depend on the browser session or a healthy EKS API.

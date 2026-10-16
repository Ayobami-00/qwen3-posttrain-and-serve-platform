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

The environment has a 24-hour maximum lifetime and an earlier budget cutoff when needed to keep the forecast below $20. The $30 AWS allowance includes builds, transfers, retention and cleanup. Estimates assume five cold GPU image downloads and no free-tier credits. Billing observations are delayed; these safeguards are not a guaranteed hard financial cap.

## Use

Open the returned dashboard URL and use the protected owner credential file. Anonymous visitors see approved summaries, settings and charts. Owner login is required for controls, raw events, downloads, Grafana and inference.

The curated recipe remains Qwen3-0.6B/GSM8K SFT with query/value LoRA targets. Clone experiments to change supported settings. Preparation identity includes pinned Hub revisions, tokenizer/template, split and sequence policy, and worker image; compatible concurrent runs share immutable data and parent artifacts.

The Go controller admits work FIFO into two shared allocation slots across training and serving. It reserves projected cost before provisioning and keeps uncertain deletions counted. Limits are $3 total Runpod spend, $1/GPU-hour and 45 minutes per allocation; recovery retains the original run's remaining runtime. A download-free BF16/Qwen3/LoRA probe runs first. One automatic infrastructure replacement may restore a committed checkpoint.

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

Bootstrap state, private evidence buckets, ECR and idle managed roles/jobs remain for recovery. Project secret containers are removed and recurring cleanup is disabled after compute absence is confirmed; an unresolved provider cleanup retains the janitor credentials. Conservative retention costs are included in the forecast. Models/checkpoints and images expire after seven days; metadata backups/reports after 30 days. Obsolete versions and unfinished uploads have bounded lifecycle rules. `destroy` deliberately does not empty evidence buckets. Preserve remote state and retained evidence before any later full bootstrap removal.

## Verification

Run the Python CPU suite, `python tests/e2e/hosted_cpu.py`, Go tests and dashboard checks. The HTTP CPU proof includes real training, worker loss, optimizer recovery, S3 publication, export parity, serving, streaming, disconnect cleanup, restart and backup/restore. It explicitly does not validate Kubernetes or GPUs. `QPS_TEST_POSTGRES_URL` enables the real migration/concurrency test; CI supplies PostgreSQL 18.6. Linux Kubernetes/browser acceptance runs before EKS creation. Live acceptance separately records overlapping GPU experiments, checkpoint replacement, Forge lineage, serving parity, measured costs and cleanup.

Reusable modules live in `terraform/core/modules/aws`; bootstrap and foundation/addons/platform roots have separate S3 state keys. Provider locks, chart versions, EKS add-on versions and container base manifests are pinned. `/docs/` remains ignored.

The provisioner includes the EC2 VPC security-group lookup required by ALB creation. EKS launch-template validation receives `RunInstances` only for the project's exact template, with instances restricted to `m7i.large`. A pinned Terraform time resource waits 30 seconds after changes to that policy before creating nodes, avoiding IAM propagation failures on a fresh install.

Forge team privacy is preserved. A private-only team receives private tracking and an owner report link; anonymous visitors continue to use native platform charts. Public report embedding is enabled only when the team's policy permits public projects.

The disposable CodeBuild Kind cluster explicitly permits its host's cgroup v1 configuration. EKS uses AL2023 nodes with cgroup v2. To run the browser proof against the live installation, set `QPS_BROWSER_URL` and `QPS_LIVE_OWNER_FILE` to the returned HTTPS URL and protected owner credential file, then run `npm --prefix web run test:e2e`. The live GPU driver is `python tests/e2e/hosted_live.py`; it persists request identities, interrupts an exactly verified owned worker at checkpoint 25, downloads only small JSON evidence, and stops its allocations in cleanup.

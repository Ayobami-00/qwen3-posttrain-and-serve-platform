# QPS Go controller

The leader-elected controller reconciles namespaced `TrainingRun`, `ComputeAllocation` and `ModelDeployment` CRDs. It writes lifecycle observations to Kubernetes before projecting status into PostgreSQL. The API's transactional outbox makes dispatch restartable.

PostgreSQL transactions own FIFO admission and cost reservations. Create intent and exact ownership are persisted before a non-idempotent Runpod request. An uncertain response is reconciled by the allocation's committed name; it never causes a blind second create. Cancellation/deadlines trigger deletion, and reservations remain until absence is confirmed. An independent Lambda uses the durable S3 allocation ledger after EKS loss.

Profiles use one external GPU per allocation or a disposable Kubernetes CPU Job for acceptance. GPU workers do not join EKS. Only the controller/janitor hold the Runpod management key. Workers receive fenced credentials and approved object access; private ECR credentials are short lived and removed during cleanup.

```bash
GOTOOLCHAIN=go1.27.2 go -C operator test ./...
GOTOOLCHAIN=go1.27.2 go -C operator build ./cmd/operator
```

Tests cover quotes, ambiguous accepted creation, restart, duplicate prevention and ownership-safe deletion using fake Kubernetes/provider interfaces. Linux acceptance runs the actual controller against Kind/PostgreSQL/S3. CRDs and namespace-scoped RBAC are in `deploy/helm/qps`.

Runpod REST v2 has no atomic maximum-price or automatic-expiry field. Admission, returned-rate verification, worker leases, runtime accounting and independent cleanup enforce the application's bounds. Provider billing evidence remains separate from estimates.

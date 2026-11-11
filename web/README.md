# QPS dashboard

Next.js 16 on Node 24.22.0; the lockfile fixes exact dependencies. Generated OpenAPI types describe submissions at compile time; the Python API validates requests at runtime. Curated view models describe approved public projections.

```bash
npm --prefix web ci
npm --prefix web run dev
npm --prefix web run typecheck
npm --prefix web run build
```

Development proxies API traffic to loopback port 8080; set `QPS_API_INTERNAL_URL` before starting/building to change it. CloudFront/ALB routes production API traffic directly to Python. Owner sessions are HttpOnly with CSRF protection. Public views are read only.

For browser acceptance, install the Python train/serve/hosted/dev dependencies, start `python tests/e2e/browser_server.py` from the repository root using that environment, and start Next.js. Then install Chromium with `npm --prefix web exec playwright install chromium` and run `npm --prefix web run test:e2e`.

That fixture uses real API/S3/CPU workers and a fake coordinator. The same browser test runs against the real Go controller in Linux Kind. It covers public boundaries, owner submission, training/configuration, artifacts, deployment, streamed inference and stop. Screenshots/results go under ignored `.qps/`.

Regenerate contracts with `python tools/generate_contracts.py` and `npm --prefix web run contracts`. Keep model weights, private download URLs and credentials out of frontend assets.

## Follow a dashboard request

[app/page.tsx](app/page.tsx) owns the experiment list/detail, creation form, model
and deployment views. Its form produces the hosted submission envelope: recipe,
compute profile, dataset-version selection and idempotency key. The effective
batch shown in the form is microbatch × accumulation × rank count.

[lib/sdk.ts](lib/sdk.ts) exposes the generated OpenAPI client used for submission.
[lib/client.ts](lib/client.ts) wraps ordinary JSON requests with same-origin
credentials, no-store caching and the session's CSRF header. Neither frontend
types nor disabled buttons replace API authorization.

The main workspace polls roughly every four seconds for refreshed state. Events,
metrics and allocations are persisted by the platform; browser React state is
only a rendering copy and can be reconstructed after reload. The backend also
offers a resumable event feed; the main workspace currently uses polling.

## Model and experiment views

[components/lifecycles.tsx](components/lifecycles.tsx) contains validation
comparisons, per-experiment Forge reports, model cards and deletion dialogs.

Validation comparisons show parent/candidate accuracy and counts, percentage-point
change, parse failures and truncations. The API marks results comparable only
when validation-manifest and generation settings agree; the comparison chart is
withheld when they differ or are unavailable.

Forge report iframes are scoped to the experiment's exact attempt IDs. Native
charts remain available independently. An unavailable embed is a synchronization
state, not proof that training stopped. Project visibility controls whether a
public embed can render; it does not make S3 model files public.

Model cards select independent model versions for serving. Experiment deletion
preserves published models/deployments; model deletion cascades to dependent
deployments; deleting a stopped deployment preserves its model.

Deletion first loads a preview and requires the resource ID. The returned
operation is polled until cleanup completes. Closing the dialog does not stop
the background operation, and hiding a row is not treated as successful cleanup.

Owner-only Grafana links filter the provisioned GPU dashboard. Allocation/rank
health and stale telemetry come from the platform API, rather than a browser
connection to the external GPU worker.

See the [acceptance guide](../tests/e2e/README.md) for browser fixture setup and
the distinct local, Kubernetes and live-provider validation gates.

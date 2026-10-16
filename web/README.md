# QPS dashboard

Next.js 16 on Node 24.22.0; the lockfile fixes exact dependencies. Generated OpenAPI types validate submissions, while curated view models describe public projections.

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

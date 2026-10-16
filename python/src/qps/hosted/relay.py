"""Authenticated Runpod-facing relay; the model process stays on loopback."""

import hmac
import threading
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from qps.hosted.http import until_disconnect


def create_relay(token, stop, output_limit=512):
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8001", timeout=90, trust_env=False
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, openapi_url=None)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(supplied, token):
            raise HTTPException(401, "Worker authentication required")
        if stop.is_set():
            raise HTTPException(503, "Worker is stopping")
        if path not in ("healthz", "readyz", "v1/models", "v1/completions", "v1/chat/completions"):
            raise HTTPException(404, "Unknown route")
        body = None
        if request.method == "POST":
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(422, "Request body must be an object")
            maximum = body.get("max_tokens", 128)
            if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum > output_limit:
                raise HTTPException(422, "Output limit exceeded")
        upstream = await until_disconnect(
            request,
            app.state.client.send(
                app.state.client.build_request(request.method, "/" + path, json=body), stream=True
            ),
        )
        if upstream.status_code != 200 or not body or not body.get("stream"):
            try:
                return Response(
                    await until_disconnect(request, upstream.aread()),
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            finally:
                await upstream.aclose()

        async def chunks():
            try:
                async for chunk in upstream.aiter_raw():
                    if stop.is_set() or await request.is_disconnected():
                        break
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            chunks(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    return app


def start_relay(token, stop, port=8000, output_limit=512):
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            create_relay(token, stop, output_limit),
            host="0.0.0.0",
            port=port,
            access_log=False,
            log_level="warning",
            timeout_graceful_shutdown=5,
        )
    )
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("Worker relay failed to start")
    return server

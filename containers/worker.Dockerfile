FROM ghcr.io/astral-sh/uv:0.11.31@sha256:ecd4de2f060c64bea0ff8ecb182ddf46ba3fcccdc8a60cfdbaf20d1a047d7437 AS uv
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc
ARG QPS_SOURCE_SHA
ENV QPS_CODE_REVISION=$QPS_SOURCE_SHA
LABEL org.opencontainers.image.revision=$QPS_SOURCE_SHA
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 ca-certificates git && rm -rf /var/lib/apt/lists/*
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app/python
COPY python/pyproject.toml python/uv.lock ./
RUN uv sync --frozen --extra train --extra serve --extra hosted --no-dev --no-install-project --no-cache
COPY python/ ./
RUN uv sync --frozen --extra train --extra serve --extra hosted --no-dev --no-cache && useradd -m -u 10001 qps && mkdir -p /workspace/qps && chown -R qps /workspace
WORKDIR /workspace/qps
USER 10001
EXPOSE 8000
ENTRYPOINT ["python", "-m", "qps.hosted.agent"]

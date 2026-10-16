FROM ghcr.io/astral-sh/uv:0.11.31@sha256:ecd4de2f060c64bea0ff8ecb182ddf46ba3fcccdc8a60cfdbaf20d1a047d7437 AS uv
# The locked PyTorch CUDA wheels supply CUDA/cuDNN user-space libraries.
# A second CUDA runtime base duplicated those libraries and inflated paid transfers.
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
ARG QPS_SOURCE_SHA
ENV QPS_CODE_REVISION=$QPS_SOURCE_SHA
LABEL org.opencontainers.image.revision=$QPS_SOURCE_SHA
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git && rm -rf /var/lib/apt/lists/*
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

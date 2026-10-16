FROM ghcr.io/astral-sh/uv:0.11.31@sha256:ecd4de2f060c64bea0ff8ecb182ddf46ba3fcccdc8a60cfdbaf20d1a047d7437 AS uv
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS api
ARG QPS_SOURCE_SHA
ENV QPS_CODE_REVISION=$QPS_SOURCE_SHA
LABEL org.opencontainers.image.revision=$QPS_SOURCE_SHA
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app/python
COPY python/pyproject.toml python/uv.lock ./
RUN uv sync --frozen --extra hosted --no-dev --no-install-project --no-cache
COPY python/ ./
RUN uv sync --frozen --extra hosted --no-dev --no-cache && useradd -m -u 10001 qps
USER 10001
EXPOSE 8080
ENTRYPOINT ["python", "-m", "qps.hosted.api"]

FROM api AS cpu-worker
USER root
# Same locked Python libraries; only the torch binary is CPU-specific in this fixture image.
RUN uv pip install --python /opt/venv/bin/python 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cpu --no-cache && \
    uv export --frozen --extra hosted --extra train --extra serve --no-dev --no-hashes --no-emit-package torch --no-emit-package triton | sed '/^nvidia-/d' > /tmp/cpu.txt && \
    uv pip install --python /opt/venv/bin/python -r /tmp/cpu.txt --no-cache && rm /tmp/cpu.txt && \
    mkdir -p /workspace/qps && chown -R qps /workspace
WORKDIR /workspace/qps
USER 10001
EXPOSE 8000
ENTRYPOINT ["python", "-m", "qps.hosted.agent"]

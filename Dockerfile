
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Keep the venv outside /app so the bind mount (and the host's macOS .venv)
# does not hide or clash with it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

CMD ["prefect", "worker", "start", "--pool", "mlops-pipeline", "--type", "process"]

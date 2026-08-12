FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

RUN uv sync \
    --froze \
    --no-dev \
    --no-install-project

COPY . .

RUN uv sync \
    --locked \
    --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD [ "uvicorn", "signalrank.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
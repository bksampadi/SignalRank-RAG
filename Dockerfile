FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
    --frozen \
    --no-dev \
    --no-install-project

ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
ENV FLASHRANK_CACHE_DIR=/app/.cache/flashrank

RUN .venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

RUN .venv/bin/python -c \
    "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/app/.cache/flashrank', max_length=512)"

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
    --locked \
    --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD [ "uvicorn", "signalrank.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
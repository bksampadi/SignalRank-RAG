# SignalRank-RAG

[![CI](https://github.com/bksampadi/SignalRank-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/bksampadi/SignalRank-RAG/actions/workflows/ci.yml)

SignalRank-RAG is an open-source framework for building and evaluating Retrieval-Augmented Generation (RAG) systems.

The project focuses on:

- Document ingestion
- Chunking strategies
- Embedding pipelines
- Retrieval systems
- Reranking
- Evaluation
- Production deployment

## Currently Implemented

- Recursive TXT and Markdown ingestion
- Structured DOCX, PPTX, and XLSX ingestion
- Portable relative source references
- Deterministic, content-sensitive document IDs
- Deterministic chunking with configurable overlap
- Preservation of document and element provenance
- SentenceTransformer embedding provider
- BM25 lexical retrieval
- Dense semantic retrieval
- Qdrant vector store integration
- Shared retrieval interface and ranked search result model
- Retrieval service supporting BM25 and dense modes
- Local FastAPI retrieval API with health and retrieval endpoints
- Structured observability and tracing with Logfire
- Reproducible dependency management with uv
- 35 automated tests

## Development Setup

SignalRank-RAG uses uv for Python environment and dependency management.

```bash
uv sync --extra eval 
uv run pytest -v
```

Run the local API:

```bash
uv run uvicorn signalrank.api.app:app --reload
```

Interactive API documentation is available locally at `/docs`.

Continuous integration currently tests Python 3.11–3.13 on Ubuntu and Python 3.11 on Windows.

## Roadmap

- [x] Project scaffold
- [x] Exception handling
- [x] Observability and tracing
- [x] Configuration management
- [x] Data ingestion
- [x] Structured Office document ingestion
- [x] Deterministic chunking
- [x] Embedding provider abstraction
- [x] SentenceTransformer embedding baseline
- [x] BM25 lexical retrieval
- [x] Qdrant vector store
- [x] Dense retrieval
- [x] Local FastAPI API
- [ ] Reranking
- [ ] Evaluation
- [ ] API deployment


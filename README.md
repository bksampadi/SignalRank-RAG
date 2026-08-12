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
- Portable relative source references
- Deterministic, content-sensitive document IDs
- Empty and unsupported-file handling
- Structured observability and tracing with Logfire
- Reproducible dependency management with uv
- 25 automated tests

## Development Setup

SignalRank-RAG uses uv for Python environment and dependency management.

```bash
uv sync --extra eval 
uv run pytest -v
```
The project currently targets Python 3.11 for local development.

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
- [ ] Vector stores
- [ ] Dense Retrieval
- [ ] Reranking
- [ ] Evaluation
- [ ] API deployment


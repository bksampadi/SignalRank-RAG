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
- 12 automated tests

## Development Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest -v
```

## Roadmap

- [x] Project scaffold
- [x] Exception handling
- [x] Logging framework
- [x] Configuration management
- [x] Data ingestion
- [ ] Chunking
- [ ] Embeddings
- [ ] Vector stores
- [ ] Retrieval
- [ ] Reranking
- [ ] Evaluation
- [ ] API deployment


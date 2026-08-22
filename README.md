# SignalRank-RAG

[![CI](https://github.com/bksampadi/SignalRank-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/bksampadi/SignalRank-RAG/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white)
[![Release](https://img.shields.io/github/v/release/bksampadi/SignalRank-RAG)](https://github.com/bksampadi/SignalRank-RAG/releases)

**Reproducible retrieval, from documents to ranked results.**

SignalRank-RAG is an open-source, production-oriented RAG system combining deterministic document processing, lexical and semantic retrieval, configurable embeddings, vector search, observability, testing, and containerized execution.


**Python · FastAPI · Qdrant · BM25 · SentenceTransformers · Gemini · Logfire · Streamlit · Docker · CI**

## Architecture


```mermaid
flowchart TB

    DOCS(["📄  Your Documents"])
    USER(["👤  Your Query"])

    ING["Ingestion"]
    CHUNK["Deterministic<br/>Chunking"]

    BMIDX[("BM25 Index")]
    EMB["✦ Embedding<br/>Provider"]
    QD[("Qdrant")]

    UI["Streamlit"]
    API["FastAPI"]
    RET{"Retrieval<br/>Service"}

    BM["BM25<br/>Retrieval"]
    DENSE["✦ Dense<br/>Retrieval"]

    RESULT(["✦  Ranked Results"])


    DOCS --> ING --> CHUNK

    CHUNK --> BMIDX
    CHUNK --> EMB --> QD


    USER --> UI --> API --> RET

    RET --> BM
    RET --> DENSE

    BMIDX -. "lexical index" .-> BM
    QD -. "semantic index" .-> DENSE

    BM --> RESULT
    DENSE --> RESULT


    classDef entry fill:#172033,stroke:#64748b,stroke-width:1.5px,color:#f8fafc;
    classDef process fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px,color:#0f172a;
    classDef ai fill:#f3e8ff,stroke:#8b5cf6,stroke-width:2px,color:#4c1d95;
    classDef store fill:#ecfdf5,stroke:#10b981,stroke-width:2px,color:#064e3b;
    classDef app fill:#eff6ff,stroke:#3b82f6,stroke-width:1.5px,color:#1e3a8a;
    classDef route fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764;
    classDef result fill:#172033,stroke:#a78bfa,stroke-width:2.5px,color:#ffffff;

    class DOCS,USER entry;
    class ING,CHUNK,BM process;
    class EMB,DENSE ai;
    class BMIDX,QD store;
    class UI,API app;
    class RET route;
    class RESULT result;
```

The same deterministic chunks feed both lexical and semantic retrieval. Dense retrieval uses a configurable embedding provider and Qdrant vector storage, while the application layer exposes retrieval through a shared FastAPI service.

### AI & Software Engineering

SignalRank-RAG separates orchestration from implementation, keeping meaningful retrieval and AI components independently replaceable.

```mermaid
flowchart LR
    P["⚙️ Retrieval Pipeline"]

    P --> E["✦ Embeddings"]
    P --> R["⌕ Retrieval"]
    P --> V["◉ Vector Store"]

    E --> E1["SentenceTransformers"]
    E --> E2["Gemini"]

    R --> R1["BM25"]
    R --> R2["Dense"]

    V --> V1["Qdrant"]

    classDef core fill:#172033,stroke:#64748b,stroke-width:2px,color:#f8fafc;
    classDef boundary fill:#f3e8ff,stroke:#8b5cf6,stroke-width:2px,color:#4c1d95;
    classDef impl fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px,color:#0f172a;

    class P core;
    class E,R,V boundary;
    class E1,E2,R1,R2,V1 impl;
```

## Highlights

- Deterministic multi-format ingestion with document and chunk provenance
- BM25 lexical and dense semantic retrieval over the same chunk corpus
- Replaceable SentenceTransformer and Gemini embedding providers
- Qdrant vector storage with dimension validation and memory/local/remote modes
- FastAPI + Streamlit application layer with distributed Logfire tracing
- 35 automated tests, multi-platform CI, locked dependencies, and containerized execution


## Technical Overview

| Area | Implementation |
| --- | --- |
| **Ingestion** | TXT, Markdown, PDF, HTML, DOCX, PPTX, and XLSX |
| **Provenance** | Portable source references and deterministic document/chunk IDs |
| **Chunking** | Configurable size and overlap |
| **Lexical retrieval** | BM25 |
| **Dense retrieval** | Embedding-based semantic search |
| **Embeddings** | Provider protocol with SentenceTransformer and Gemini implementations |
| **Local embedding model** | `sentence-transformers/all-mpnet-base-v2` |
| **Cloud embedding model** | `gemini-embedding-2` |
| **Vector storage** | Qdrant with dimension validation |
| **Configuration** | Typed dataclasses backed by YAML |
| **API** | FastAPI |
| **UI** | Streamlit |
| **Observability** | Logfire distributed tracing |
| **Testing** | 35 automated tests |
| **CI** | Python 3.11–3.13 on Ubuntu and Windows |
| **Packaging** | `uv` with locked dependencies |
| **Runtime** | Docker and Docker Compose |

## Quick Start

The simplest way to run SignalRank-RAG is with Docker Compose.

```bash
docker compose up --build
```

Then open:

- **Streamlit UI:** `http://localhost:8501`
- **FastAPI documentation:** `http://localhost:8000/docs`
- **Health endpoint:** `http://localhost:8000/health`

The Streamlit UI and FastAPI backend run as separate services.

## Development

SignalRank-RAG uses [`uv`](https://docs.astral.sh/uv/) for Python environment and dependency management.

Install dependencies:

```bash
uv sync --extra eval
```

Run the test suite:

```bash
uv run --locked python -m pytest -v
```

Run FastAPI:

```bash
uv run uvicorn signalrank.api.main:app --reload
```

Run Streamlit in a second terminal:

```bash
uv run streamlit run src/signalrank/ui/app.py
```

By default, Streamlit expects the retrieval API at:

```text
http://127.0.0.1:8000
```

Override this with:

```text
SIGNALRANK_API_URL
```

## Configuration

Application behavior is configured through `configs/config.yaml`.

For example:

```yaml
chunking:
  chunk_size: 1000
  chunk_overlap: 200

embedding:
  provider: sentence_transformer
  model_name: sentence-transformers/all-mpnet-base-v2

retrieval:
  top_k: 5

qdrant:
  mode: local
  collection_name: signalrank_finewiki
  path: data/qdrant
```


Qdrant supports three execution modes:

```text
memory  → ephemeral in-memory storage
local   → persistent local storage
remote  → remote Qdrant / Qdrant Cloud
```

Deployment-specific endpoints and credentials are supplied through environment variables:

```text
GEMINI_API_KEY
QDRANT_URL
QDRANT_API_KEY
LOGFIRE_TOKEN
```
Do not store secrets in `config.yaml`.

## Observability

SignalRank-RAG uses Logfire for structured and distributed tracing across the retrieval path.

```text
Streamlit
    ↓
HTTP
    ↓
FastAPI
    ↓
Retrieval
    ↓
Embedding
    ↓
Qdrant
```


Manual spans record operational metadata including retrieval mode, embedding provider and dimension, collection, workload size, and execution timing. Raw query and document text is not intentionally attached to manual telemetry.

Logfire is optional. SignalRank-RAG can run without Logfire credentials.

## Project Structure

```text
SignalRank-RAG/
├── .github/
│   └── workflows/
│       └── ci.yml              Multi-platform CI
│
├── configs/                    YAML application configuration
├── data/                       Local datasets and vector-store data
├── scripts/                    Development and pipeline utilities
│
├── src/
│   └── signalrank/
│       ├── api/                FastAPI application
│       ├── components/
│       │   ├── chunking/       Deterministic document chunking
│       │   ├── data_ingestion/ Document discovery and loaders
│       │   ├── embeddings/     Embedding providers and factory
│       │   ├── retrieval/      BM25 and dense retrieval
│       │   └── vector_store/   Qdrant integration
│       ├── config/             Typed configuration loading
│       ├── pipelines/          Ingestion and indexing workflows
│       ├── services/           Application-level services
│       └── ui/                 Streamlit interface
│
├── tests/                      Automated test suite
│
├── .dockerignore
├── .gitignore
├── .python-version
├── compose.yaml
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

## Current Status

`v0.1.0` established the initial retrieval foundation with deterministic ingestion, chunking, lexical and dense retrieval, Qdrant vector storage, API and UI layers, distributed tracing, automated testing, CI, and containerized execution.

Current development extends that baseline with configurable embedding providers, Gemini embeddings, model-aware vector dimensions, and configurable Qdrant execution modes.

## Roadmap

- [ ] Hybrid retrieval
- [ ] Reranking
- [ ] Evaluation framework
- [ ] Retrieval diagnostics
- [ ] Embedding model comparison
- [ ] Public deployment
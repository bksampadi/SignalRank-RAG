<p align="center">
  <a href="https://signalrank-ui.icybeach-127c89e3.northeurope.azurecontainerapps.io">
    <img src="docs/assets/live_demo_cta.svg" alt="Try SignalRank live" width="1200">
  </a>
</p>

# SignalRank-RAG

**Reproducible retrieval, reranking, and grounded responses.**

SignalRank-RAG is an open-source RAG application built with **Python, FastAPI, LangGraph, and Qdrant**. It combines deterministic ingestion, hybrid retrieval, optional FlashRank reranking, evidence-backed generation, and automated cloud deployment.

[![CI](https://github.com/bksampadi/SignalRank-RAG/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/bksampadi/SignalRank-RAG/actions/workflows/ci_cd.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.13-506C8A?logo=python&logoColor=white)
[![React / TypeScript](https://img.shields.io/badge/Console-React%20%2B%20TypeScript-3178C6?logo=react&logoColor=white)](https://github.com/bksampadi/signalrank-console)
[![Release](https://img.shields.io/github/v/release/bksampadi/SignalRank-RAG?label=release&color=8A6A46)](https://github.com/bksampadi/SignalRank-RAG/releases)
![Docker](https://img.shields.io/badge/Docker-containerized-47708A?logo=docker&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-vector%20database-875566?logo=qdrant&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-Container%20Apps-4D7188?logo=microsoftazure&logoColor=white)
[![Cloudflare](https://img.shields.io/badge/Cloudflare-Pages-F38020?logo=cloudflare&logoColor=white)](https://github.com/bksampadi/signalrank-console)

The live **Streamlit demo** exposes BM25, dense, and hybrid retrieval with inspectable evidence and relevance feedback. A separate **[React/TypeScript Console](https://github.com/bksampadi/signalrank-console)** provides a Cloudflare Pages frontend and gateway.

---

## System Architecture

<p align="center">
  <img height="26" src="https://img.shields.io/badge/AI%20%2F%20ML-665191?style=flat-square" alt="AI / ML">
  <img height="26" src="https://img.shields.io/badge/Retrieval%20%2F%20RAG-3F718C?style=flat-square" alt="Retrieval / RAG">
  <img height="26" src="https://img.shields.io/badge/Agentic%20Orchestration-A95D45?style=flat-square" alt="Agentic Orchestration">
  <img height="26" src="https://img.shields.io/badge/Platform%20%2F%20Software%20Engineering-557567?style=flat-square" alt="Platform / Software Engineering">
</p>

Three views show distinct lifecycles: **request processing**, **corpus/index preparation**, and **delivery/runtime**. Colors indicate engineering responsibilities across all three.

### 1. Request processing

```mermaid
flowchart TB
    USER(["👤 User"])

    subgraph UX["🖥️ Interfaces"]
        direction TB
        STREAM["Streamlit demo<br/>Azure Container Apps"]
        CONSOLE["React + TypeScript Console<br/>Cloudflare Pages"]
        GATE["🛡️ Cloudflare gateway<br/>Turnstile · quota"]
        CONSOLE --> GATE
    end

    GUARD["🔐 FastAPI<br/>service token · daily quota"]
    RET["POST /retrieve"]
    CHAT["POST /chat"]
    SEARCH["🔎 SearchService<br/>BM25 · dense · hybrid/RRF<br/>optional FlashRank"]
    GRAPH["🕸️ LangGraph<br/>route · retrieve · respond"]
    LLM["✨ LLMService<br/>Groq default · OpenRouter optional"]
    QD[("🗄️ Qdrant")]
    OUT["✅ Answer or ranked evidence"]
    OBS["📈 Logfire"]

    USER --> STREAM
    USER --> CONSOLE
    STREAM -->|server-side token| GUARD
    GATE -.->|API integration| GUARD
    GUARD --> RET
    GUARD --> CHAT
    RET --> SEARCH
    CHAT -->|other requests| GRAPH
    CHAT -->|greeting shortcut| OUT
    GRAPH --> SEARCH
    GRAPH -.->|LLM calls| LLM
    QD -. semantic results .-> SEARCH
    SEARCH -->|direct retrieval| OUT
    GRAPH -->|chat response| OUT
    GRAPH -.->|LLM down: evidence only| OUT
    GUARD -. traces .-> OBS

    classDef ai fill:#665191,stroke:#A99CE8,stroke-width:1.5px,color:#ffffff
    classDef rag fill:#3F718C,stroke:#82A8CF,stroke-width:1.5px,color:#ffffff
    classDef agentic fill:#A95D45,stroke:#D6A75C,stroke-width:1.5px,color:#ffffff
    classDef platform fill:#557567,stroke:#72AA9E,stroke-width:1.5px,color:#ffffff
    classDef neutral fill:#475569,stroke:#94A3B8,stroke-width:1.5px,color:#ffffff
    class STREAM,CONSOLE,GATE,GUARD,RET,CHAT,OBS platform
    class SEARCH,QD rag
    class GRAPH agentic
    class LLM ai
    class USER,OUT neutral
```

`/retrieve` invokes `SearchService` directly; `/chat` uses LangGraph for conversation, grounded synthesis, or evidence-only responses. Both reuse the same retrieval and optional reranking pipeline. Common greetings bypass the graph.

| `/chat` mode | Behaviour |
| --- | --- |
| `auto` | Structured LLM routing between conversation and retrieval |
| `synthesis` | Retrieve and generate a grounded answer |
| `evidence` | Return ranked evidence without LLM synthesis |

**Reliability:** The responder abstains when evidence is insufficient. Router failures fall back to retrieval and the evidence responder; generation failures return evidence without further synthesis. Groq is the default configured provider, with OpenRouter as an alternative—not automatic cross-provider failover.

**Security:** `/chat` and `/retrieve` use `X-SignalRank-Service-Token` and share an **in-process limit of 100 requests per UTC day per process** (HTTP 429). The counter resets on restart; it is not distributed. `/health` is unauthenticated, and browser code never receives the service token.

### 2. Corpus and index lifecycle

```mermaid
flowchart TB
    DOCS["📄 Source documents<br/>PDF · Office · HTML · text"]
    HF["🤗 Hugging Face Hub<br/>all-mpnet-base-v2 model"]

    subgraph BUILD["📥 1 · Index build"]
        direction TB
        LOAD["Load + deterministic chunking"] --> EMB["🧠 Embed document chunks"]
        EMB --> QD[("🗄️ Qdrant<br/>persisted vectors + payloads")]
    end

    subgraph STARTUP["⚡ 2 · API startup"]
        direction TB
        ATTACH["Connect to existing<br/>Qdrant collection"]
        RELOAD["Reload source corpus<br/>reconstruct identical chunks"]
        MODEL["Initialize embedding model<br/>for query encoding"]
        BM["🔤 BM25<br/>in memory"]
        IDS["Chunk-ID lookup"]
        RELOAD --> BM
        RELOAD --> IDS
    end

    SEARCH["🔎 SearchService<br/>lexical · semantic · hybrid/RRF"]

    DOCS --> LOAD
    HF -.->|model weights| EMB
    QD -->|persisted index| ATTACH
    DOCS -.->|re-read at startup| RELOAD
    HF -.->|same model| MODEL
    ATTACH --> SEARCH
    BM --> SEARCH
    IDS --> SEARCH
    MODEL --> SEARCH
    QD -. semantic results .-> SEARCH

    classDef ai fill:#665191,stroke:#A99CE8,stroke-width:1.5px,color:#ffffff
    classDef rag fill:#3F718C,stroke:#82A8CF,stroke-width:1.5px,color:#ffffff
    classDef neutral fill:#475569,stroke:#94A3B8,stroke-width:1.5px,color:#ffffff
    class HF,EMB,MODEL ai
    class LOAD,QD,ATTACH,RELOAD,BM,IDS,SEARCH rag
    class DOCS neutral
```

**Index build → API startup** is the normal lifecycle: `scripts/index_corpus.py` stores chunk vectors in Qdrant; the API connects to that existing collection and **re-reads the source corpus** to rebuild BM25 and its chunk-ID lookup. Dashed arrows distinguish reused inputs from the primary flow. `recreate_collection: true` can instead rebuild Qdrant during startup.

The production configuration uses Hugging Face's `sentence-transformers/all-mpnet-base-v2` model for both document and query embeddings (downloaded when absent from cache). Separately, SignalRank has an **optional FineWiki streaming source** via `HuggingFaceFW/finewiki`; the default production corpus is local files, not a live FineWiki dataset.

### 3. Delivery and runtime

```mermaid
flowchart TB

    subgraph BACKEND["🐍 SignalRank-RAG · Python"]
        direction TB

        SRC["GitHub · SignalRank-RAG"]
        CI["⚙️ GitHub Actions<br/>Ruff · Pyright · pytest<br/>Docker smoke test"]
        GHCR["📦 GHCR<br/>Commit-SHA image"]
        OIDC["🔑 Azure OIDC<br/>Deployment authorization"]

        subgraph AZURE["☁️ Azure Container Apps"]
            direction TB
            API["⚡ FastAPI<br/>/retrieve · /chat"]
            STREAM["🖥️ Streamlit<br/>Live demo"]
        end

        VERIFY["✅ Deployment verification<br/>Image SHA · UI reachability"]

        SRC --> CI
        CI -->|main| GHCR
        CI -->|main| OIDC

        GHCR --> API
        GHCR --> STREAM
        OIDC --> API
        OIDC --> STREAM

        STREAM -->|API requests| API
        API --> VERIFY
        STREAM --> VERIFY
    end

    subgraph FRONTEND["⚛️ SignalRank Console · React / TypeScript"]
        direction TB

        REPO["GitHub · signalrank-console"]
        PAGES["☁️ Cloudflare Pages<br/>React · TypeScript · Vite"]
        GATEWAY["🛡️ Pages Functions<br/>Turnstile · D1 quota"]

        REPO --> PAGES
        PAGES --> GATEWAY
    end

    GATEWAY -.->|HTTPS /retrieve<br/>server-side token| API

    API -.->|traces| LOG["📈 Logfire"]
    SRC -.->|manual validation| K8S["☸️ Kubernetes<br/>kind"]

    classDef platform fill:#557567,stroke:#72AA9E,stroke-width:1.5px,color:#ffffff
    classDef neutral fill:#475569,stroke:#94A3B8,stroke-width:1.5px,color:#ffffff

    class CI,GHCR,OIDC,API,STREAM,VERIFY,PAGES,GATEWAY,LOG,K8S platform
    class SRC,REPO neutral
```
---

## Project Structure

```text
SignalRank-RAG/
├── .github/workflows/       # CI/CD
├── benchmarks/retrieval/    # corpus, queries, ground truth
├── configs/                 # application and benchmark settings
├── deploy/kubernetes/       # kind-validated runtime manifests
├── scripts/                 # indexing, benchmarking, local demo
├── src/signalrank/
│   ├── agents/              # LangGraph routing and responders
│   ├── api/                 # FastAPI endpoints and request checks
│   ├── components/          # loaders, chunking, embeddings, retrieval, ranking
│   ├── config/              # typed YAML configuration
│   ├── evaluation/          # retrieval metrics
│   ├── observability/       # Logfire integration
│   ├── pipelines/           # indexing and retrieval orchestration
│   ├── services/            # search and LLM service boundaries
│   └── ui/                  # Streamlit and feedback
├── tests/
├── compose.yaml
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

---

## Retrieval and Reranking

- **BM25:** Lexical matching for exact terms.
- **Dense:** Semantic retrieval using embeddings in Qdrant.
- **Hybrid:** Reciprocal rank fusion of lexical and dense rankings.
- **FlashRank:** Optional second-stage reranking of retrieved candidates.

Retrieval determines the candidate set; reranking reorders it. Both are exposed through replaceable service boundaries.

### Adversarial retrieval benchmark

The reproducible benchmark contains **39 documents and 30 labelled queries**. Results measure **first-stage retrieval**, before reranking or answer generation.

<p align="center">
  <img src="docs/assets/signalrank_benchmark.svg" alt="SignalRank retrieval benchmark" width="900">
</p>

| Mode | Hit@1 | MRR@5 |
| --- | ---: | ---: |
| BM25 | 0.433 | 0.590 |
| Dense | **0.567** | **0.756** |
| Hybrid (RRF) | 0.467 | 0.653 |

Dense retrieval achieves **Hit@5 = 1.000** on this corpus, but BM25 finds top-1 matches dense retrieval misses (`BM25 only: 2`; `dense only: 6`). Their top-1 oracle reaches **0.633**; baseline hybrid fusion still underperforms dense retrieval. This motivates better fusion and reranking rather than assuming hybrid search must win.

<details>
<summary><strong>Reproduce the benchmark</strong></summary>

```bash
uv sync --extra eval
uv run --locked python scripts/index_corpus.py --config configs/benchmark.yaml --recreate
uv run --locked python scripts/benchmark_retrieval.py
```

The benchmark configuration pins the `sentence-transformers/all-mpnet-base-v2` model, corpus, chunking, and retrieval settings. The model must be available locally or downloaded on first use.

</details>

---

## API

| Endpoint | Purpose | Authentication |
| --- | --- | --- |
| `GET /health` | Liveness check | None |
| `POST /retrieve` | BM25, dense, or hybrid results; optional reranking | Service token |
| `POST /chat` | Routing, grounded synthesis, or evidence-only response | Service token |

`/chat` supports `auto`, `synthesis`, and `evidence` response modes. The API accepts `X-SignalRank-Service-Token` for protected endpoints.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-SignalRank-Service-Token: signalrank-local-dev" \
  -d '{"query":"What caused the extinction of the dinosaurs?","mode":"hybrid","top_k":5}'
```

The example token is for local development; production credentials must be supplied through secrets.

---

## Quick Start

```bash
docker compose up --build
```

- **Streamlit:** http://localhost:8501
- **FastAPI docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

For the benchmark-backed local demo on Windows:

```cmd
scripts\run_demo.cmd
```

<details>
<summary><strong>Development and configuration</strong></summary>

The project uses [`uv`](https://docs.astral.sh/uv/) with a committed lockfile.

```bash
uv sync --extra eval
uv run --locked python -m pytest -v
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyright
```

Run the services directly:

```bash
uv run uvicorn signalrank.api.main:app --reload
uv run streamlit run src/signalrank/ui/app.py
```

YAML configuration covers ingestion, embeddings, retrieval, fusion, reranking, Qdrant, response modes, and LLM providers. Qdrant supports in-memory, local, or remote operation. Deployment credentials come from environment variables or managed secrets, not committed configuration.

Common environment variables:

```text
GROQ_API_KEY
GEMINI_API_KEY
OPENROUTER_API_KEY
QDRANT_URL
QDRANT_API_KEY
LOGFIRE_TOKEN
SIGNALRANK_API_URL
SIGNALRANK_SERVICE_TOKEN
```

</details>

---

## Evaluation and Observability

The Streamlit UI collects **Relevant / Not relevant** judgements with query and result metadata for offline retrieval diagnostics. Feedback does not change rankings online.

**Logfire** traces API requests, retrieval, embeddings, reranking, routing, LLM calls, provider failures, and feedback events. Some evaluation logs may include submitted queries or result metadata; do not enter sensitive information into the public demo. Logfire is optional locally.

---

## Kubernetes

Kubernetes manifests under [`deploy/kubernetes`](deploy/kubernetes)
are validated locally with `kind`, including Deployment, ClusterIP Service,
health probes, resource limits, HorizontalPodAutoscaler, Secrets,
ConfigMaps, and Kustomize.

Azure Container Apps remains the production runtime.

---

## Roadmap

**Implemented:** Deterministic ingestion; BM25, dense, and hybrid retrieval; optional FlashRank reranking; LangGraph routing, evidence-only fallback, and basic abstention; adversarial evaluation; relevance feedback; API token/quota; Azure CI/CD; Cloudflare Pages Console deployment; kind-validated Kubernetes manifests.

**Next:** Reranker benchmarks and ranking diagnostics; improved fusion; stronger evidence-sufficiency and abstention calibration; larger retrieval benchmarks; retrieval-quality gates in CI; richer routing policies.

---

## Cite

```bibtex
@software{signalrank_rag,
  author = {Bharath Sampadi},
  title = {SignalRank-RAG: Reproducible Retrieval, Reranking, and Grounded Responses},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/bksampadi/SignalRank-RAG}
}
```

## License

MIT License. See [`LICENSE`](LICENSE).

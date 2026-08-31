import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import logfire
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, status

from signalrank.api.schemas import (
    ChatRequest,
    ChatResponse,
    RetrieveRequest,
    RetrieveResponse,
    SearchResultResponse,
)
from signalrank.components.retrieval.evidence import (
    has_sufficient_evidence,
)
from signalrank.config.configuration import ConfigurationManager
from signalrank.constants import CONFIG_FILE_PATH
from signalrank.pipelines.indexing_pipeline import IndexingPipeline
from signalrank.pipelines.ranking_pipeline import RankingPipeline
from signalrank.pipelines.retrieval_pipeline import RetrievalPipeline
from signalrank.services.ranking_service import RankingService
from signalrank.services.retrieval_service import RetrievalService
from signalrank.services.search_service import SearchService


def verify_service_token(
    token: str | None,
    expected: str,
) -> None:

    if token is None or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )


def logfire_request_attributes(
    _request: Request | WebSocket,
    attributes: dict[str, Any],
) -> dict[str, Any] | None:
    if attributes["errors"]:
        return {
            "errors": attributes["errors"],
        }

    return {}

def get_conversation_response(
    query: str,
) -> str | None:
    normalized = query.strip().lower()

    if normalized in {"hi", "hello", "hey"}:
        return "Hello! Ask me something about the demo knowledge base."

    if normalized in {
        "who are you?",
        "who are you",
    }:
        return (
            "I'm SignalRank-RAG, an evidence-first retrieval system. "
            "I retrieve and rerank relevant passages from the demo corpus."
        )

    if normalized in {
        "what can you do?",
        "what can you do",
    }:
        return (
            "I can search the demo corpus using sparse, dense, or hybrid "
            "retrieval and show you the ranked evidence behind the answer."
        )

    if normalized in {
        "what's up?",
        "what's up",
        "whats up?",
        "whats up",
    }:
        return "Not much 😄 Ask me something from the demo knowledge base."

    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    config_filepath = Path(
        os.getenv(
            "SIGNALRANK_CONFIG",
            str(CONFIG_FILE_PATH),
        )
    )

    service_token = os.getenv("SIGNALRANK_SERVICE_TOKEN")

    if not service_token:
        raise RuntimeError("SIGNALRANK_SERVICE_TOKEN is required.")

    app.state.service_token = service_token

    config = ConfigurationManager(config_filepath).load()
    app.state.config = config

    if config.qdrant.recreate_collection:
        IndexingPipeline(
            config_filepath=config_filepath,
        ).run()

    bm25, dense, hybrid = RetrievalPipeline(
        config_filepath=config_filepath,
    ).build()

    retrieval_service = RetrievalService(
        retrievers={
            "bm25": bm25,
            "dense": dense,
            "hybrid": hybrid,
        }
    )

    rerankers = RankingPipeline(
        config_filepath=config_filepath,
    ).build()

    ranking_service = RankingService(
        rerankers=rerankers,
    )

    search_service = SearchService(
        retrieval_service=retrieval_service,
        ranking_service=ranking_service,
        ranking_mode=config.ranking.provider,
        ranking_enabled=config.ranking.enabled,
        candidate_multiplier=config.ranking.candidate_multiplier,
    )

    app.state.search_service = search_service

    yield


app = FastAPI(
    title="SignalRank-RAG",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
)
def retrieve(
    payload: RetrieveRequest,
    request: Request,
    service_token: str | None = Header(
        default=None,
        alias="X-SignalRank-Service-Token",
    ),
) -> RetrieveResponse:
    verify_service_token(
        service_token,
        request.app.state.service_token,
    )

    service: SearchService = request.app.state.search_service

    with logfire.span(
        "search corpus evidence",
        retrieval_mode=payload.mode,
        top_k=payload.top_k,
        query_length=len(payload.query),
    ):
        results = service.search(
            query=payload.query,
            retrieval_mode=payload.mode,
            top_k=payload.top_k,
        )

    return RetrieveResponse(
        query=payload.query,
        mode=payload.mode,
        results=[
            SearchResultResponse(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                text=result.text,
                score=result.score,
                rank=result.rank,
                source_path=result.source_path,
                metadata=result.metadata,
            )
            for result in results
        ],
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    payload: ChatRequest,
    request: Request,
    service_token: str | None = Header(
        default=None, alias="X-SignalRank-Service-Token"
    ),
) -> ChatResponse:
    verify_service_token(
        service_token,
        request.app.state.service_token,
    )

    conversation_answer = get_conversation_response(
        payload.query,
    )

    if conversation_answer is not None:
        return ChatResponse(
            query=payload.query,
            route="conversation",
            response_mode=payload.response_mode,
            answer=conversation_answer,
            results=[],
        )

    service: SearchService = request.app.state.search_service

    with logfire.span(
        "chat evidence query",
        retrieval_mode=payload.mode,
        top_k=payload.top_k,
        query_length=len(payload.query),
        response_mode=payload.response_mode,
    ):
        results = service.search(
            query=payload.query,
            retrieval_mode=payload.mode,
            top_k=payload.top_k,
        )

    if not has_sufficient_evidence(results):
        answer = (
            "I couldn't find reliable evidence for that "
            "in the demo corpus. "
            "Try a topic such as dinosaurs, Mars, vaccines, batteries, "
            "Python, retrieval, or mythology."
        )
    else:
        answer = results[0].text

    return ChatResponse(
        query=payload.query,
        route="retrieval",
        response_mode=payload.response_mode,
        answer=answer,
        results=[
            SearchResultResponse(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                text=result.text,
                score=result.score,
                rank=result.rank,
                source_path=result.source_path,
                metadata=result.metadata,
            )
            for result in results
        ],
    )


logfire.configure(
    send_to_logfire="if-token-present",
    distributed_tracing=True,
)

logfire.instrument_fastapi(
    app,
    request_attributes_mapper=logfire_request_attributes,
)

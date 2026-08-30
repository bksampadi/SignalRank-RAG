import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import logfire
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, status
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter

from signalrank.agents.graph import build_agent_graph
from signalrank.api.schemas import (
    ChatRequest,
    ChatResponse,
    RetrieveRequest,
    RetrieveResponse,
    SearchResultResponse,
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


def get_agent_graph(request: Request):
    graph = request.app.state.agent_graph

    if graph is None:
        config = request.app.state.config

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_model = os.getenv("OPENROUTER_MODEL")

        if not openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required.")

        if not openrouter_model:
            raise RuntimeError("OPENROUTER_MODEL is required.")
        
        llm = ChatOpenRouter(
            model=openrouter_model,
            max_retries=config.llm.max_retries,
            temperature=0,
        )

        graph = build_agent_graph(
            search_service=request.app.state.search_service,
            llm=llm,
        )

        request.app.state.agent_graph = graph

    return graph


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
    app.state.agent_graph = None

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

    graph = get_agent_graph(request)

    with logfire.span(
        "agent query",
        retrieval_mode=payload.mode,
        top_k=payload.top_k,
        query_length=len(payload.query),
    ):
        state = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=payload.query,
                    )
                ],
                "current_query": payload.query,
                "retrieval_mode": payload.mode,
                "top_k": payload.top_k,
            }
        )

        route = state.get("route")
        answer = state.get("final_answer")

        if route is None:
            raise RuntimeError("Agent graph completed without setting route.")

        if answer is None:
            raise RuntimeError("Agent graph completed without setting final_answer.")

    results = state.get(
        "search_results",
        [],
    )

    return ChatResponse(
        query=payload.query,
        route=route,
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

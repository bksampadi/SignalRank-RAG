import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import logfire
from fastapi import FastAPI, Request, WebSocket

from signalrank.api.schemas import (
    RetrieveRequest,
    RetrieveResponse,
    SearchResultResponse,
)
from signalrank.constants import CONFIG_FILE_PATH
from signalrank.pipelines.retrieval_pipeline import RetrievalPipeline
from signalrank.services.retrieval_service import RetrievalService


def logfire_request_attributes(
    request: Request | WebSocket,
    attributes: dict[str, Any],
) -> dict[str, Any] | None:
    if attributes["errors"]:
        return {
            "errors": attributes["errors"],
        }

    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_filepath = Path(
        os.getenv(
            "SIGNALRANK_CONFIG",
            str(CONFIG_FILE_PATH),
        )
    )

    bm25, dense, hybrid = RetrievalPipeline(
        config_filepath=config_filepath,
    ).build()

    app.state.retrieval_service = RetrievalService(
        retrievers={
            "bm25": bm25,
            "dense": dense,
            "hybrid": hybrid,
        }
    )

    yield


app = FastAPI(
    title="SignalRank-RAG",
    version="0.1.0",
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
) -> RetrieveResponse:
    service: RetrievalService = request.app.state.retrieval_service

    with logfire.span(
        "retrieve corpus evidence",
        retrieval_mode=payload.mode,
        top_k=payload.top_k,
        query_length=len(payload.query),
    ):
        results = service.retrieve(
            query=payload.query,
            mode=payload.mode,
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


logfire.configure(
    send_to_logfire="if-token-present",
    distributed_tracing=True,
)

logfire.instrument_fastapi(
    app,
    request_attributes_mapper=logfire_request_attributes,
)

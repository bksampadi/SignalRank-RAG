from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

import logfire

from signalrank.api.schemas import(
    RetrieveRequest,
    RetrieveResponse,
    SearchResultResponse,
)
from signalrank.pipelines.retrieval_pipeline import RetrievalPipeline
from signalrank.services.retrieval_service import RetrievalService

def logfire_request_attributes(
    request: Request,
    attributes: dict,
) -> dict:
    if attributes["errors"]:
        return {
            "errors": attributes["errors"],
        }

    return {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    bm25, dense = RetrievalPipeline().build()

    app.state.retrieval_service = RetrievalService(
        retrievers={
            "bm25": bm25,
            "dense": dense,
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
    service: RetrievalService = (
        request.app.state.retrieval_service
    )

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
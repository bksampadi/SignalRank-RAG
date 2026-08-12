from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from signalrank.api.schemas import(
    RetrieveRequest,
    RetrieveResponse,
    SearchResultResponse,
)
from signalrank.pipelines.retrieval_pipeline import RetrievalPipeline
from signalrank.services.retrieval_service import RetrievalService


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

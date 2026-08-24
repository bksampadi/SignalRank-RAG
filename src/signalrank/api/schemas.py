from typing import Literal

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal[
        "bm25",
        "dense",
        "hybrid",
    ] = "dense"
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResultResponse(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    source_path: str
    metadata: dict[str, object]


class RetrieveResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResultResponse]

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RetrieveRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
    )
    mode: Literal[
        "bm25",
        "dense",
        "hybrid",
    ] = "dense"
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("query must not be blank")

        return value


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


class ChatRequest(RetrieveRequest):
    pass


class ChatResponse(BaseModel):
    query: str
    route: Literal[
        "conversation",
        "retrieval",
    ]
    answer: str
    results: list[SearchResultResponse]

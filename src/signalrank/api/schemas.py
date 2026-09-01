from pydantic import BaseModel, Field, field_validator

from signalrank.types import (
    EffectiveResponseMode,
    FallbackReason,
    ResponseMode,
    RetrievalMode,
    Route,
)


class RetrieveRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
    )
    mode: RetrievalMode = "dense"
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
    mode: RetrievalMode
    results: list[SearchResultResponse]


class ChatRequest(RetrieveRequest):
    response_mode: ResponseMode = "auto"


class ChatResponse(BaseModel):
    query: str
    route: Route
    response_mode: ResponseMode
    effective_response_mode: EffectiveResponseMode

    fallback_reason: FallbackReason | None = None

    answer: str
    results: list[SearchResultResponse]

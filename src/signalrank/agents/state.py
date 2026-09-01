from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from signalrank.components.retrieval.result import SearchResult
from signalrank.types import (
    EffectiveResponseMode,
    FallbackReason,
    ResponseMode,
    RetrievalMode,
    Route,
)


class AgentState(TypedDict):
    """
    Shared state passed between SignalRank agent nodes.

    """

    messages: Annotated[list[BaseMessage], add_messages]

    current_query: str
    retrieval_mode: RetrievalMode
    response_mode: ResponseMode
    top_k: int

    route: NotRequired[Route]
    search_results: NotRequired[list[SearchResult]]
    final_answer: NotRequired[str]

    effective_response_mode: NotRequired[EffectiveResponseMode]

    fallback_reason: NotRequired[FallbackReason]

    plan: NotRequired[list[str]]
    current_step: NotRequired[int]
    tool_results: NotRequired[list[Any]]

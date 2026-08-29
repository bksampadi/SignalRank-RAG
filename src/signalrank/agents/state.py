from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from signalrank.components.retrieval.result import SearchResult

Route = Literal[
    "conversation",
    "retrieval",
]

RetrievalMode = Literal[
    "bm25",
    "dense",
    "hybrid",
]


class AgentState(TypedDict):
    """
    Shared state passed between SignalRank agent nodes.

    """

    messages: Annotated[list[BaseMessage], add_messages]

    current_query: str
    retrieval_mode: RetrievalMode
    top_k: int

    route: NotRequired[Route]

    search_results: NotRequired[list[SearchResult]]

    final_answer: NotRequired[str]

    plan: NotRequired[list[str]]

    current_step: NotRequired[int]

    tool_results: NotRequired[list[Any]]

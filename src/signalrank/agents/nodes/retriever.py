from langchain_core.runnables import Runnable, RunnableLambda

from signalrank.agents.state import AgentState
from signalrank.components.retrieval.result import SearchResult
from signalrank.services.search_service import SearchService


def make_retriever_node(
    search_service: SearchService,
) -> Runnable[
    AgentState,
    dict[str, list[SearchResult]],
]:
    """
    Create a retrieval node bound to SignalRank's retrieval service.
    """

    def retriever_node(
        state: AgentState,
    ) -> dict[str, list[SearchResult]]:
        results = search_service.search(
            query=state["current_query"],
            retrieval_mode=state["retrieval_mode"],
            top_k=state["top_k"],
        )

        return {
            "search_results": results,
        }

    return RunnableLambda(retriever_node)

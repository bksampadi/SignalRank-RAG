from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from signalrank.agents.nodes.responder import make_responder_node
from signalrank.agents.nodes.retriever import make_retriever_node
from signalrank.agents.nodes.router import make_router_node
from signalrank.agents.state import AgentState, Route
from signalrank.services.search_service import SearchService


def route_after_router(
    state: AgentState,
) -> Route:
    route = state.get("route")

    if route is None:
        raise RuntimeError("Router did not set a route.")

    return route


def build_agent_graph(
    *,
    search_service: SearchService,
    llm: BaseChatModel,
):
    builder = StateGraph(AgentState)

    builder.add_node(
        "router",
        make_router_node(llm),
    )

    builder.add_node(
        "retriever",
        make_retriever_node(search_service),
    )

    builder.add_node(
        "responder",
        make_responder_node(llm),
    )

    builder.add_edge(
        START,
        "router",
    )

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "conversation": "responder",
            "retrieval": "retriever",
        },
    )

    builder.add_edge(
        "retriever",
        "responder",
    )

    builder.add_edge(
        "responder",
        END,
    )

    return builder.compile()

from typing import Literal

from langgraph.graph import END, START, StateGraph

from signalrank.agents.nodes.responder import (
    make_evidence_responder_node,
    make_responder_node,
)
from signalrank.agents.nodes.retriever import make_retriever_node
from signalrank.agents.nodes.router import make_router_node
from signalrank.agents.state import AgentState, Route
from signalrank.services.llm_service import LLMService
from signalrank.services.search_service import SearchService

EntryRoute = Literal[
    "router",
    "retriever",
]

ResponseRoute = Literal[
    "responder",
    "evidence_responder",
]


def route_from_start(
    state: AgentState,
) -> EntryRoute:
    response_mode = state["response_mode"]

    if response_mode == "auto":
        return "router"

    return "retriever"


def route_after_retrieval(
    state: AgentState,
) -> ResponseRoute:
    if state["response_mode"] == "evidence":
        return "evidence_responder"

    return "responder"


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
    llm_service: LLMService,
):
    builder = StateGraph(AgentState)

    builder.add_node(
        "router",
        make_router_node(llm_service),
    )

    builder.add_node(
        "retriever",
        make_retriever_node(search_service),
    )

    builder.add_node(
        "responder",
        make_responder_node(llm_service),
    )

    builder.add_node(
        "evidence_responder",
        make_evidence_responder_node(),
    )

    builder.add_conditional_edges(
        START,
        route_from_start,
        {
            "router": "router",
            "retriever": "retriever",
        },
    )

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "conversation": "responder",
            "retrieval": "retriever",
        },
    )

    builder.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {
            "responder": "responder",
            "evidence_responder": "evidence_responder",
        },
    )

    builder.add_edge(
        "evidence_responder",
        END,
    )

    return builder.compile()

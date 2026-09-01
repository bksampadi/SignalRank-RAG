import logfire
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from signalrank.agents.state import AgentState
from signalrank.prompts.rag_prompts import ROUTER_SYSTEM_PROMPT
from signalrank.services.llm_service import (
    LLMService,
    LLMUnavailableError,
)
from signalrank.types import Route


class RouteDecision(BaseModel):
    route: Route


def make_router_node(
    llm_service: LLMService,
) -> Runnable[AgentState, dict[str, object]]:
    """
    Create a router node bound to an LLM.
    """

    def router_node(
        state: AgentState,
    ) -> dict[str, object]:
        try:
            decision = llm_service.invoke_structured(
                [
                    SystemMessage(
                        content=ROUTER_SYSTEM_PROMPT,
                    ),
                    HumanMessage(
                        content=state["current_query"],
                    ),
                ],
                RouteDecision,
            )

            return {
                "route": decision.route,
            }
        except LLMUnavailableError:
            logfire.warn("LLM router unavailable; switching to retrieval")

            return {"route": "retrieval", "fallback_reason": "llm_unavailable"}

    return RunnableLambda(router_node)

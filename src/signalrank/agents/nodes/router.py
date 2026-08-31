from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from signalrank.agents.state import AgentState, Route
from signalrank.prompts.rag_prompts import ROUTER_SYSTEM_PROMPT
from signalrank.services.llm_service import LLMService


class RouteDecision(BaseModel):
    route: Route


def make_router_node(
    llm_service: LLMService,
) -> Runnable[AgentState, dict[str, Route]]:
    """
    Create a router node bound to an LLM.
    """

    def router_node(
        state: AgentState,
    ) -> dict[str, Route]:
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

    return RunnableLambda(router_node)

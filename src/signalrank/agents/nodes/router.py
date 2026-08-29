from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from signalrank.agents.state import AgentState, Route
from signalrank.prompts.rag_prompts import ROUTER_SYSTEM_PROMPT


class RouteDecision(BaseModel):
    route: Route


def make_router_node(
    llm: BaseChatModel,
) -> Runnable[AgentState, dict[str, Route]]:
    """
    Create a router node bound to an LLM.
    """

    router_llm = llm.with_structured_output(RouteDecision)

    def router_node(
        state: AgentState,
    ) -> dict[str, Route]:
        decision = cast(
            RouteDecision,
            router_llm.invoke(
                [
                    SystemMessage(
                        content=ROUTER_SYSTEM_PROMPT,
                    ),
                    HumanMessage(
                        content=state["current_query"],
                    ),
                ]
            ),
        )

        return {
            "route": decision.route,
        }

    return RunnableLambda(router_node)

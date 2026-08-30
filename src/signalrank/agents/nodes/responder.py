from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableLambda

from signalrank.agents.state import AgentState
from signalrank.components.retrieval.result import SearchResult
from signalrank.prompts.rag_prompts import (
    CONVERSATION_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
)

# Conservative initial abstention threshold.
# Calibrate against benchmark score distributions before treating as final.
MIN_EVIDENCE_SCORE = 1e-4


def _has_sufficient_evidence(
    results: list[SearchResult],
) -> bool:
    if not results:
        return False

    return max(result.score for result in results) >= MIN_EVIDENCE_SCORE


def _format_context(
    results: list[SearchResult],
) -> str:
    sections = []

    for index, result in enumerate(results, start=1):
        sections.append(
            f"[{index}]\n"
            f"Source: {result.source_path}\n"
            f"Chunk ID: {result.chunk_id}\n"
            f"{result.text}"
        )

    return "\n\n".join(sections)


def make_responder_node(
    llm: BaseChatModel,
) -> Runnable[
    AgentState,
    dict[str, object],
]:
    """
    Create the final response-generation node.
    """

    def responder_node(
        state: AgentState,
    ) -> dict[str, object]:

        route = state.get("route")

        if route is None:
            raise RuntimeError("Agent route was not set before responder execution.")

        if route == "conversation":
            response = llm.invoke(
                [
                    SystemMessage(
                        content=CONVERSATION_SYSTEM_PROMPT,
                    ),
                    *state["messages"],
                ]
            )

        else:
            results = state.get(
                "search_results",
                [],
            )

            if not _has_sufficient_evidence(results):
                response = AIMessage(
                    content=(
                        "I couldn't find reliable evidence for that in the demo corpus. "
                        "I can help with topics such as dinosaurs, Mars, vaccines, "
                        "batteries, Python, retrieval, and mythology."
                    )
                )

            else:
                context = _format_context(results)

                """
                Keep previous conversation turns, but replace the
                current query with an evidence-grounded prompt.
                """

                history = state["messages"][:-1]

                response = llm.invoke(
                    [
                        SystemMessage(
                            content=RAG_SYSTEM_PROMPT,
                        ),
                        *history,
                        HumanMessage(
                            content=(
                                f"Question:\n"
                                f"{state['current_query']}\n\n"
                                f"Retrieved evidence:\n"
                                f"{context}"
                            )
                        ),
                    ]
                )

        if not isinstance(response.content, str):
            raise TypeError(
                f"Expected text response, got {type(response.content).__name__}"
            )

        result = {
            "messages": [response],
            "final_answer": response.content,
        }

        return result

    return RunnableLambda(responder_node)

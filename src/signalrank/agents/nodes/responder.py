from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableLambda

from signalrank.agents.state import AgentState
from signalrank.components.retrieval.evidence import (
    has_sufficient_evidence,
)
from signalrank.components.retrieval.result import SearchResult
from signalrank.prompts.rag_prompts import (
    CONVERSATION_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
)
from signalrank.services.llm_service import (
    LLMService,
    LLMUnavailableError,
)

# Conservative initial abstention threshold.
# Calibrate against benchmark score distributions before treating as final.


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


def _format_evidence_response(
    results: list[SearchResult],
    *,
    fallback: bool = False,
) -> str:
    if not has_sufficient_evidence(results):
        return "I couldn't find reliable evidence for that in the demo corpus."

    count = len(results)
    noun = "passage" if count == 1 else "passages"

    if fallback:
        return (
            "LLM synthesis is temporarily unavailable, "
            f"so SignalRank switched to evidence mode. "
            f"Retrieved {count} ranked evidence {noun}."
        )

    return f"Retrieved {count} ranked evidence {noun}. No LLM synthesis was used."


def make_evidence_responder_node() -> Runnable[
    AgentState,
    dict[str, object],
]:
    """
    Create a deterministic response node that exposes retrieval
    results without calling an LLM.
    """

    def evidence_responder_node(
        state: AgentState,
    ) -> dict[str, object]:
        results = state.get(
            "search_results",
            [],
        )

        fallback = state.get("fallback_reason") == "llm_unavailable"

        answer = _format_evidence_response(
            results,
            fallback=fallback,
        )

        response = AIMessage(
            content=answer,
        )

        return {
            "messages": [response],
            "final_answer": answer,
            "effective_response_mode": "evidence",
        }

    return RunnableLambda(evidence_responder_node)


def make_responder_node(
    llm_service: LLMService,
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

        fallback_reason = None

        if route == "conversation":
            try:
                response = llm_service.invoke(
                    [
                        SystemMessage(
                            content=CONVERSATION_SYSTEM_PROMPT,
                        ),
                        *state["messages"],
                    ]
                )
                effective_response_mode = "conversation"

            except LLMUnavailableError:
                response = AIMessage(
                    content=(
                        "Conversational generation is temporarily unavailable. "
                        "Corpus retrieval remains available."
                    )
                )

                effective_response_mode = "conversation"
                fallback_reason = "llm_unavailable"

        else:
            results = state.get(
                "search_results",
                [],
            )

            if not has_sufficient_evidence(results):
                response = AIMessage(
                    content=(
                        "I couldn't find reliable evidence for that in the demo corpus. "
                        "I can help with topics such as dinosaurs, Mars, vaccines, "
                        "batteries, Python, retrieval, and mythology."
                    )
                )

                effective_response_mode = "evidence"

            else:
                context = _format_context(results)

                # Keep previous conversation turns, but replace the
                # current query with an evidence-grounded prompt.

                history = state["messages"][:-1]

                try:
                    response = llm_service.invoke(
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

                    effective_response_mode = "synthesis"
                    fallback_reason = None

                except LLMUnavailableError:
                    response = AIMessage(
                        content=_format_evidence_response(
                            results,
                            fallback=True,
                        )
                    )

                    effective_response_mode = "evidence"
                    fallback_reason = "llm_unavailable"

        if not isinstance(response.content, str):
            raise TypeError(
                f"Expected text response, got {type(response.content).__name__}"
            )

        result: dict[str, object] = {
            "messages": [response],
            "final_answer": response.content,
            "effective_response_mode": effective_response_mode,
        }

        if fallback_reason is not None:
            result["fallback_reason"] = fallback_reason

        return result

    return RunnableLambda(responder_node)

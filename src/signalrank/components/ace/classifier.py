from collections.abc import Sequence
from typing import Literal, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from signalrank.components.ace.models import (
    EvidenceStance,
    StanceEvidence,
)
from signalrank.components.retrieval.result import SearchResult
from signalrank.prompts.ace_prompts import ACE_STANCE_SYSTEM_PROMPT
from signalrank.services.llm_service import LLMService


class ACEClassificationError(RuntimeError):
    """Raised when stance classification returns an invalid result."""


class PassageClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    stance: Literal[
        "supports",
        "contradicts",
        "qualifies",
        "irrelevant",
    ]


class ClassificationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classifications: list[PassageClassification]


class StanceClassifier(Protocol):
    def classify(
        self,
        *,
        query: str,
        candidate_answer: str,
        evidence: Sequence[SearchResult],
    ) -> tuple[StanceEvidence, ...]: ...


class LLMStanceClassifier:
    def __init__(
        self,
        llm_service: LLMService,
    ) -> None:
        self._llm_service = llm_service

    def classify(
        self,
        *,
        query: str,
        candidate_answer: str,
        evidence: Sequence[SearchResult],
    ) -> tuple[StanceEvidence, ...]:
        if not evidence:
            return ()

        evidence_text = "\n\n".join(
            (f"chunk_id: {result.chunk_id}\npassage:\n{result.text}")
            for result in evidence
        )

        user_prompt = (
            f"User query:\n{query}\n\n"
            f"Candidate answer:\n{candidate_answer}\n\n"
            f"Retrieved passages:\n{evidence_text}"
        )

        response = self._llm_service.invoke_structured(
            [
                SystemMessage(content=ACE_STANCE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ],
            ClassificationBatch,
        )

        return self._map_response(
            evidence=evidence,
            response=response,
        )

    @staticmethod
    def _map_response(
        *,
        evidence: Sequence[SearchResult],
        response: ClassificationBatch,
    ) -> tuple[StanceEvidence, ...]:
        expected_ids = [result.chunk_id for result in evidence]
        returned_ids = [item.chunk_id for item in response.classifications]

        if len(returned_ids) != len(set(returned_ids)):
            raise ACEClassificationError("Classifier returned duplicate chunk IDs.")

        if set(returned_ids) != set(expected_ids):
            raise ACEClassificationError(
                "Classifier output does not match supplied evidence."
            )

        classifications = {
            item.chunk_id: EvidenceStance(item.stance)
            for item in response.classifications
        }

        return tuple(
            StanceEvidence(
                result=result,
                stance=classifications[result.chunk_id],
            )
            for result in evidence
        )

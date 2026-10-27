import pytest

from signalrank.components.ace.classifier import (
    ACEClassificationError,
    ClassificationBatch,
    LLMStanceClassifier,
    PassageClassification,
)
from signalrank.components.ace.models import EvidenceStance
from signalrank.components.retrieval.result import SearchResult


def make_result(
    chunk_id: str,
    rank: int,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=f"Passage {chunk_id}",
        score=1.0,
        rank=rank,
        source_path="test.txt",
    )


def test_maps_classifier_response_to_original_evidence_order() -> None:
    evidence = [
        make_result("a", 1),
        make_result("b", 2),
    ]

    response = ClassificationBatch(
        classifications=[
            PassageClassification(
                chunk_id="b",
                stance="contradicts",
            ),
            PassageClassification(
                chunk_id="a",
                stance="supports",
            ),
        ]
    )

    result = LLMStanceClassifier._map_response(
        evidence=evidence,
        response=response,
    )

    assert [item.result.chunk_id for item in result] == ["a", "b"]
    assert result[0].stance is EvidenceStance.SUPPORTS
    assert result[1].stance is EvidenceStance.CONTRADICTS


def test_rejects_missing_classifier_result() -> None:
    evidence = [
        make_result("a", 1),
        make_result("b", 2),
    ]

    response = ClassificationBatch(
        classifications=[
            PassageClassification(
                chunk_id="a",
                stance="supports",
            ),
        ]
    )

    with pytest.raises(
        ACEClassificationError,
        match="does not match supplied evidence",
    ):
        LLMStanceClassifier._map_response(
            evidence=evidence,
            response=response,
        )


def test_rejects_duplicate_classifier_result() -> None:
    evidence = [
        make_result("a", 1),
        make_result("b", 2),
    ]

    response = ClassificationBatch(
        classifications=[
            PassageClassification(
                chunk_id="a",
                stance="supports",
            ),
            PassageClassification(
                chunk_id="a",
                stance="irrelevant",
            ),
        ]
    )

    with pytest.raises(
        ACEClassificationError,
        match="duplicate chunk IDs",
    ):
        LLMStanceClassifier._map_response(
            evidence=evidence,
            response=response,
        )

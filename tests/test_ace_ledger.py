from signalrank.components.ace.ledger import EvidenceLedger
from signalrank.components.ace.models import (
    EvidenceStance,
    StanceEvidence,
)
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


def test_ledger_detects_answer_changing_evidence() -> None:
    evidence = [
        StanceEvidence(
            make_result("a", 1),
            EvidenceStance.SUPPORTS,
        ),
        StanceEvidence(
            make_result("b", 2),
            EvidenceStance.CONTRADICTS,
        ),
        StanceEvidence(
            make_result("c", 3),
            EvidenceStance.QUALIFIES,
        ),
    ]

    ledger = EvidenceLedger.from_evidence(evidence)

    assert ledger.evaluated_depth == 3
    assert ledger.supports == 1
    assert ledger.contradicts == 1
    assert ledger.qualifies == 1
    assert ledger.irrelevant == 0
    assert ledger.answer_changing_evidence_found is True


def test_ledger_does_not_flag_clean_evidence() -> None:
    evidence = [
        StanceEvidence(
            make_result("a", 1),
            EvidenceStance.SUPPORTS,
        ),
        StanceEvidence(
            make_result("b", 2),
            EvidenceStance.IRRELEVANT,
        ),
    ]

    ledger = EvidenceLedger.from_evidence(evidence)

    assert ledger.evaluated_depth == 2
    assert ledger.supports == 1
    assert ledger.contradicts == 0
    assert ledger.qualifies == 0
    assert ledger.irrelevant == 1
    assert ledger.answer_changing_evidence_found is False
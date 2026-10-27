from collections.abc import Sequence

from signalrank.components.ace.models import (
    EvidenceStance,
    StanceEvidence,
)
from signalrank.components.retrieval.result import SearchResult
from signalrank.evaluation.ace_eval import (
    ACEEvalCase,
    evaluate_ace_case,
)


class FakeStanceClassifier:
    def __init__(
        self,
        stances: Sequence[EvidenceStance],
    ) -> None:
        self._stances = tuple(stances)

    def classify(
        self,
        *,
        query: str,
        candidate_answer: str,
        evidence: Sequence[SearchResult],
    ) -> tuple[StanceEvidence, ...]:
        del query
        del candidate_answer

        return tuple(
            StanceEvidence(
                result=result,
                stance=stance,
            )
            for result, stance in zip(
                evidence,
                self._stances,
                strict=True,
            )
        )


def make_result(
    chunk_id: str,
    rank: int,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id="policy",
        text=f"Passage {chunk_id}",
        score=1.0,
        rank=rank,
        source_path="policy.txt",
    )


def test_ace_eval_reports_perfect_classification() -> None:
    expected = (
        EvidenceStance.SUPPORTS,
        EvidenceStance.CONTRADICTS,
        EvidenceStance.QUALIFIES,
        EvidenceStance.IRRELEVANT,
    )

    case = ACEEvalCase(
        query="Can employees work remotely?",
        candidate_answer="Employees may work remotely.",
        evidence=(
            make_result("support", 1),
            make_result("contradict", 2),
            make_result("qualify", 3),
            make_result("irrelevant", 4),
        ),
        expected_stances=expected,
    )

    classifier = FakeStanceClassifier(expected)

    result = evaluate_ace_case(
        classifier,
        case,
    )

    assert result.total == 4
    assert result.correct == 4
    assert result.stance_accuracy == 1.0
    assert result.false_flags == 0
    assert result.false_flag_rate == 0.0


def test_ace_eval_counts_false_flags() -> None:
    case = ACEEvalCase(
        query="Can employees work remotely?",
        candidate_answer="Employees may work remotely.",
        evidence=(
            make_result("support", 1),
            make_result("irrelevant", 2),
        ),
        expected_stances=(
            EvidenceStance.SUPPORTS,
            EvidenceStance.IRRELEVANT,
        ),
    )

    classifier = FakeStanceClassifier(
        (
            EvidenceStance.SUPPORTS,
            EvidenceStance.QUALIFIES,
        )
    )

    result = evaluate_ace_case(
        classifier,
        case,
    )

    assert result.stance_accuracy == 0.5
    assert result.false_flags == 1
    assert result.safe_cases == 2
    assert result.false_flag_rate == 0.5


def test_ace_eval_rejects_mismatched_case_lengths() -> None:
    try:
        ACEEvalCase(
            query="Question",
            candidate_answer="Answer",
            evidence=(make_result("a", 1),),
            expected_stances=(),
        )
    except ValueError as exc:
        assert str(exc) == ("Evidence and expected stances must have the same length.")
    else:
        raise AssertionError("Expected ACEEvalCase to reject invalid input.")

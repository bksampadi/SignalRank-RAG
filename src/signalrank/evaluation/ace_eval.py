from collections.abc import Sequence
from dataclasses import dataclass

from signalrank.components.ace.classifier import StanceClassifier
from signalrank.components.ace.models import EvidenceStance
from signalrank.components.retrieval.result import SearchResult


@dataclass(frozen=True)
class ACEEvalCase:
    query: str
    candidate_answer: str
    evidence: tuple[SearchResult, ...]
    expected_stances: tuple[EvidenceStance, ...]

    def __post_init__(self) -> None:
        if len(self.evidence) != len(self.expected_stances):
            raise ValueError(
                "Evidence and expected stances must have the same length."
            )


@dataclass(frozen=True)
class ACEEvalResult:
    expected: tuple[EvidenceStance, ...]
    predicted: tuple[EvidenceStance, ...]

    @property
    def total(self) -> int:
        return len(self.expected)

    @property
    def correct(self) -> int:
        return sum(
            expected is predicted
            for expected, predicted in zip(
                self.expected,
                self.predicted,
                strict=True,
            )
        )

    @property
    def stance_accuracy(self) -> float:
        if self.total == 0:
            return 0.0

        return self.correct / self.total

    @property
    def false_flags(self) -> int:
        safe_stances = {
            EvidenceStance.SUPPORTS,
            EvidenceStance.IRRELEVANT,
        }

        answer_changing_stances = {
            EvidenceStance.CONTRADICTS,
            EvidenceStance.QUALIFIES,
        }

        return sum(
            expected in safe_stances
            and predicted in answer_changing_stances
            for expected, predicted in zip(
                self.expected,
                self.predicted,
                strict=True,
            )
        )

    @property
    def safe_cases(self) -> int:
        return sum(
            stance
            in {
                EvidenceStance.SUPPORTS,
                EvidenceStance.IRRELEVANT,
            }
            for stance in self.expected
        )

    @property
    def false_flag_rate(self) -> float:
        if self.safe_cases == 0:
            return 0.0

        return self.false_flags / self.safe_cases


def evaluate_ace_case(
    classifier: StanceClassifier,
    case: ACEEvalCase,
) -> ACEEvalResult:
    classifications = classifier.classify(
        query=case.query,
        candidate_answer=case.candidate_answer,
        evidence=case.evidence,
    )

    predicted = tuple(
        classification.stance
        for classification in classifications
    )

    if len(predicted) != len(case.expected_stances):
        raise ValueError(
            "Classifier returned a different number of classifications."
        )

    return ACEEvalResult(
        expected=case.expected_stances,
        predicted=predicted,
    )


def evaluate_ace_cases(
    classifier: StanceClassifier,
    cases: Sequence[ACEEvalCase],
) -> ACEEvalResult:
    expected: list[EvidenceStance] = []
    predicted: list[EvidenceStance] = []

    for case in cases:
        result = evaluate_ace_case(
            classifier,
            case,
        )

        expected.extend(result.expected)
        predicted.extend(result.predicted)

    return ACEEvalResult(
        expected=tuple(expected),
        predicted=tuple(predicted),
    )
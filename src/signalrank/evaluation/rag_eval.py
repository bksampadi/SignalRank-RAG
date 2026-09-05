from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnswerabilityMetrics:
    total: int
    correct_answers: int
    correct_abstentions: int
    false_answers: int
    false_abstentions: int


def answerability_metrics(
    should_answer: list[bool],
    answered: list[bool],
) -> AnswerabilityMetrics:
    """
    Evaluate answer-vs-abstain decisions.

    '''false_answers''' means the system answered a case labelled unanswerable.
    It does not yet measure semantic correctness of generated answer text.
    """

    if len(should_answer) != len(answered):
        raise ValueError("Inputs must have equal lengths.")

    correct_answers = 0
    correct_abstentions = 0
    false_answers = 0
    false_abstentions = 0

    for expected, predicted in zip(should_answer, answered, strict=True):
        if expected and predicted:
            correct_answers += 1
        elif not expected and not predicted:
            correct_abstentions += 1
        elif not expected and predicted:
            false_answers += 1
        else:
            false_abstentions += 1

    return AnswerabilityMetrics(
        total=len(should_answer),
        correct_answers=correct_answers,
        correct_abstentions=correct_abstentions,
        false_answers=false_answers,
        false_abstentions=false_abstentions,
    )


def decisions_at_threshold(
    scores: list[float],
    threshold: float,
) -> list[bool]:
    return [score >= threshold for score in scores]

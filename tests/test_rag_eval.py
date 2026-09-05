from signalrank.evaluation.rag_eval import (
    answerability_metrics,
    decisions_at_threshold,
)


def test_answerability_metrics() -> None:
    metrics = answerability_metrics(
        should_answer=[True, True, False, False], answered=[True, False, True, False]
    )

    assert metrics.correct_answers == 1
    assert metrics.correct_abstentions == 1
    assert metrics.false_answers == 1
    assert metrics.false_abstentions == 1


def test_decisions_at_threshold() -> None:
    decisions = decisions_at_threshold(
        scores=[0.8, 0.3, 0.6],
        threshold=0.5,
    )

    assert decisions == [True, False, True]

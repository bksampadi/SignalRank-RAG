from signalrank.evaluation.rag_eval import answerability_metrics


def test_answerability_metrics() -> None:
    metrics = answerability_metrics(
        should_answer=[True, True, False, False], answered=[True, False, True, False]
    )

    assert metrics.correct_answers == 1
    assert metrics.correct_abstentions == 1
    assert metrics.false_answers == 1
    assert metrics.false_abstentions == 1

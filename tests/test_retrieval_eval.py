import pytest

from signalrank.evaluation.retrieval_eval import (
    deduplicate_ranked_ids,
    evaluate_retrieval,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

RETRIEVED_IDS = [
    "chunk_8",
    "chunk_3",
    "chunk_14",
    "chunk_2",
    "chunk_21",
]

RELEVANT_IDS = {
    "chunk_3",
    "chunk_14",
    "chunk_19",
}


def test_precision_at_k() -> None:
    assert precision_at_k(
        RETRIEVED_IDS,
        RELEVANT_IDS,
        k=3,
    ) == pytest.approx(2 / 3)


def test_recall_at_k() -> None:
    assert recall_at_k(
        RETRIEVED_IDS,
        RELEVANT_IDS,
        k=3,
    ) == pytest.approx(2 / 3)


def test_reciprocal_rank_at_k() -> None:
    assert reciprocal_rank_at_k(
        RETRIEVED_IDS,
        RELEVANT_IDS,
        k=3,
    ) == pytest.approx(0.5)


def test_metrics_when_no_relevant_item_is_retrieved() -> None:
    retrieved_ids = ["a", "b", "c"]
    relevant_ids = {"x"}

    assert (
        hit_rate_at_k(
            retrieved_ids,
            relevant_ids,
            k=3,
        )
        == 0.0
    )

    assert (
        precision_at_k(
            retrieved_ids,
            relevant_ids,
            k=3,
        )
        == 0.0
    )

    assert (
        recall_at_k(
            retrieved_ids,
            relevant_ids,
            k=3,
        )
        == 0.0
    )

    assert (
        reciprocal_rank_at_k(
            retrieved_ids,
            relevant_ids,
            k=3,
        )
        == 0.0
    )


@pytest.mark.parametrize(
    "metric",
    [
        hit_rate_at_k,
        precision_at_k,
        recall_at_k,
        reciprocal_rank_at_k,
    ],
)
def test_metrics_reject_non_positive_k(metric) -> None:
    with pytest.raises(
        ValueError,
        match="k must be greater than 0",
    ):
        metric(
            RETRIEVED_IDS,
            RELEVANT_IDS,
            k=0,
        )


def test_hit_rate_at_k() -> None:
    assert (
        hit_rate_at_k(
            RETRIEVED_IDS,
            RELEVANT_IDS,
            k=3,
        )
        == 1.0
    )


def test_evaluate_retrieval() -> None:
    retrieved_ids_by_query = [
        ["a", "b", "c"],
        ["x", "y", "z"],
    ]

    relevant_ids_by_query = [
        {"b"},
        {"x"},
    ]

    result = evaluate_retrieval(
        retrieved_ids_by_query,
        relevant_ids_by_query,
        k=3,
    )

    assert result.hit_rate_at_k == 1.0
    assert result.precision_at_k == pytest.approx(1 / 3)
    assert result.recall_at_k == 1.0
    assert result.mrr_at_k == pytest.approx(0.75)


def test_evaluate_retrieval_rejects_mismatched_query_sets() -> None:
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        evaluate_retrieval(
            [["a", "b"]],
            [{"a"}, {"b"}],
            k=2,
        )


def test_evaluate_retrieval_requires_queries() -> None:
    with pytest.raises(
        ValueError,
        match="at least one query",
    ):
        evaluate_retrieval(
            [],
            [],
            k=5,
        )


def test_deduplicate_ranked_ids_preserves_order() -> None:
    ids = [
        "doc_a",
        "doc_a",
        "doc_b",
        "doc_a",
        "doc_c",
    ]

    assert deduplicate_ranked_ids(ids) == [
        "doc_a",
        "doc_b",
        "doc_c",
    ]

from dataclasses import dataclass
from math import log2
from statistics import fmean


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Return 1.0 if at least one relevant item appears
    in the top-k retrieved results.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = retrieved_ids[:k]

    return float(any(item_id in relevant_ids for item_id in top_k))


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate the fraction of top-k retrieved items
    that are relevant.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_retrieved = sum(item_id in relevant_ids for item_id in top_k)

    return relevant_retrieved / len(top_k)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate the fraction of all relevant items
    retrieved within the top-k results.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    relevant_retrieved = sum(item_id in relevant_ids for item_id in top_k)

    return relevant_retrieved / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate the reciprocal rank of the first
    relevant item within the top-k results.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    for rank, item_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if item_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
    relevance_scores: dict[str, float] | None = None,
) -> float:
    """
    Calculate normalized discounted cumulative gain at k.

    Binary relevance is used by default. Optional relevance scores
    allow graded relevance judgments.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        return 0.0

    if relevance_scores is not None:
        unknown_ids = set(relevance_scores) - relevant_ids

        if unknown_ids:
            raise ValueError("relevance_scores must only contain relevant IDs")

        if any(score < 0 for score in relevance_scores.values()):
            raise ValueError("relevance scores must be non-negative")

    def relevance(item_id: str) -> float:
        if item_id not in relevant_ids:
            return 0.0

        if relevance_scores is None:
            return 1.0

        return relevance_scores.get(
            item_id,
            1.0,
        )

    def gain(score: float) -> float:
        return (2.0**score) - 1.0

    dcg = sum(
        gain(relevance(item_id)) / log2(rank + 1)
        for rank, item_id in enumerate(
            retrieved_ids[:k],
            start=1,
        )
    )

    ideal_relevances = sorted(
        (relevance(item_id) for item_id in relevant_ids),
        reverse=True,
    )[:k]

    idcg = sum(
        gain(score) / log2(rank + 1)
        for rank, score in enumerate(
            ideal_relevances,
            start=1,
        )
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def deduplicate_ranked_ids(
    ids: list[str],
) -> list[str]:
    """
    Remove duplicate IDs while preserving ranking order.
    """
    return list(dict.fromkeys(ids))


@dataclass(frozen=True)
class QueryEvaluationResult:
    """
    Retrieval metrics and ranking evidence for one query.
    """

    retrieved_ids: tuple[str, ...]
    relevant_ids: frozenset[str]
    k: int

    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank_at_k: float
    ndcg_at_k: float


def evaluate_query(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
    relevance_scores: dict[str, float] | None = None,
) -> QueryEvaluationResult:
    """
    Evaluate retrieval performance for one query.
    """
    return QueryEvaluationResult(
        retrieved_ids=tuple(retrieved_ids),
        relevant_ids=frozenset(relevant_ids),
        k=k,
        hit_rate_at_k=hit_rate_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        precision_at_k=precision_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        recall_at_k=recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        reciprocal_rank_at_k=reciprocal_rank_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        ndcg_at_k=ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            k,
            relevance_scores=relevance_scores,
        ),
    )


@dataclass(frozen=True)
class RetrievalEvaluation:
    """
    Aggregate retrieval metrics across a query set.
    """

    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float = 0.0


def evaluate_retrieval(
    retrieved_ids_by_query: list[list[str]],
    relevant_ids_by_query: list[set[str]],
    k: int,
    relevance_scores_by_query: (list[dict[str, float] | None] | None) = None,
) -> RetrievalEvaluation:
    """
    Evaluate retrieval performance across multiple queries.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if len(retrieved_ids_by_query) != len(relevant_ids_by_query):
        raise ValueError("retrieved and relevant query sets must have equal length")

    if not retrieved_ids_by_query:
        raise ValueError("at least one query is required")

    if relevance_scores_by_query is None:
        normalized_relevance_by_scores: list[dict[str, float] | None] = [
            None for _ in retrieved_ids_by_query
        ]

    else:
        if len(relevance_scores_by_query) != len(retrieved_ids_by_query):
            raise ValueError("relevance score query sets must have equal length")

        normalized_relevance_by_scores = relevance_scores_by_query

    query_results = [
        evaluate_query(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
            relevance_scores=relevance_scores,
        )
        for (
            retrieved_ids,
            relevant_ids,
            relevance_scores,
        ) in zip(
            retrieved_ids_by_query,
            relevant_ids_by_query,
            normalized_relevance_by_scores,
            strict=True,
        )
    ]

    return aggregate_results(query_results)


def aggregate_results(
    query_results: list[QueryEvaluationResult],
) -> RetrievalEvaluation:
    """
    Aggregate per-query retrieval results into mean metrics.
    """
    if not query_results:
        raise ValueError("at least one query result is required")

    return RetrievalEvaluation(
        hit_rate_at_k=fmean(result.hit_rate_at_k for result in query_results),
        precision_at_k=fmean(result.precision_at_k for result in query_results),
        recall_at_k=fmean(result.recall_at_k for result in query_results),
        mrr_at_k=fmean(result.reciprocal_rank_at_k for result in query_results),
        ndcg_at_k=fmean(result.ndcg_at_k for result in query_results),
    )

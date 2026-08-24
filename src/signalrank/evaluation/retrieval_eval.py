from dataclasses import dataclass
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


def deduplicate_ranked_ids(
    ids: list[str],
) -> list[str]:
    """
    Remove duplicate IDs while preserving ranking order.
    """
    return list(dict.fromkeys(ids))


@dataclass(frozen=True)
class RetrievalEvaluation:
    """
    Aggregate retrieval metrics across a query set.
    """

    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float


def evaluate_retrieval(
    retrieved_ids_by_query: list[list[str]],
    relevant_ids_by_query: list[set[str]],
    k: int,
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

    hit_rates = []
    precisions = []
    recalls = []
    reciprocal_ranks = []

    for retrieved_ids, relevant_ids in zip(
        retrieved_ids_by_query,
        relevant_ids_by_query,
        strict=True,
    ):
        hit_rates.append(
            hit_rate_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )
        )

        precisions.append(
            precision_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )
        )

        recalls.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )
        )

    return RetrievalEvaluation(
        hit_rate_at_k=fmean(hit_rates),
        precision_at_k=fmean(precisions),
        recall_at_k=fmean(recalls),
        mrr_at_k=fmean(reciprocal_ranks),
    )

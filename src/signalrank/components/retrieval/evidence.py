from signalrank.components.retrieval.result import SearchResult

MIN_EVIDENCE_SCORE = 1e-4


def has_sufficient_evidence(
    results: list[SearchResult],
) -> bool:
    if not results:
        return False

    return max(result.score for result in results) >= MIN_EVIDENCE_SCORE

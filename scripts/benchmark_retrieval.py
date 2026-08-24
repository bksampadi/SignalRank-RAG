import json
from pathlib import Path

from signalrank.components.chunking.chunking import DocumentChunker
from signalrank.config.configuration import ConfigurationManager
from signalrank.constants import BENCHMARK_CONFIG_FILE_PATH
from signalrank.evaluation.retrieval_eval import (
    deduplicate_ranked_ids,
    evaluate_retrieval,
)
from signalrank.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from signalrank.pipelines.retrieval_pipeline import RetrievalPipeline
from signalrank.services.retrieval_service import RetrievalService

QUERY_FILE_PATH = "data/benchmarks/retrieval/queries.json"


def load_benchmark_queries() -> list[dict]:
    with open(
        QUERY_FILE_PATH,
        encoding="utf-8",
    ) as file:
        return json.load(file)


def resolve_ground_truth(
    benchmark_queries: list[dict],
) -> list[dict]:
    config = ConfigurationManager(BENCHMARK_CONFIG_FILE_PATH).load()

    documents = DataIngestionPipeline(
        config_filepath=BENCHMARK_CONFIG_FILE_PATH,
    ).run()

    chunks = DocumentChunker(
        config=config.chunking,
    ).chunk_documents(documents)

    resolved_queries = []

    for benchmark_query in benchmark_queries:
        relevant_filenames = set(benchmark_query["relevant_docs"])

        relevant_doc_ids = {
            chunk.doc_id
            for chunk in chunks
            if Path(chunk.source_path).name in relevant_filenames
        }

        evidence = benchmark_query["evidence"]

        relevant_chunk_ids = {
            chunk.chunk_id
            for chunk in chunks
            if (
                Path(chunk.source_path).name in relevant_filenames
                and evidence in chunk.text
            )
        }

        if not relevant_doc_ids:
            raise ValueError(
                f"Could not resolve relevant document for {benchmark_query['id']}"
            )

        if not relevant_chunk_ids:
            raise ValueError(
                f"Could not resolve relevant evidence for {benchmark_query['id']}"
            )

        resolved_queries.append(
            {
                **benchmark_query,
                "relevant_doc_ids": relevant_doc_ids,
                "relevant_chunk_ids": relevant_chunk_ids,
            }
        )

    return resolved_queries


def evaluate_mode(
    retrieval_service: RetrievalService,
    mode: str,
    benchmark_queries: list[dict],
) -> tuple[
    dict[str, float],
    list[str | None],
    list[list[str]],
]:

    retrieved_chunks_by_query: list[list[str]] = []
    retrieved_docs_by_query: list[list[str]] = []

    relevant_chunks_by_query: list[set[str]] = []
    relevant_docs_by_query: list[set[str]] = []

    top_docs_by_query: list[str | None] = []

    for benchmark_query in benchmark_queries:
        query = benchmark_query["query"]

        results = retrieval_service.retrieve(
            query=query,
            mode=mode,
            top_k=5,
        )

        chunk_ids = [result.chunk_id for result in results]

        doc_ids = deduplicate_ranked_ids([result.doc_id for result in results])

        top_docs_by_query.append(doc_ids[0] if doc_ids else None)

        retrieved_chunks_by_query.append(chunk_ids)
        retrieved_docs_by_query.append(doc_ids)

        relevant_chunks_by_query.append(benchmark_query["relevant_chunk_ids"])
        relevant_docs_by_query.append(benchmark_query["relevant_doc_ids"])

    chunk_at_1 = evaluate_retrieval(
        retrieved_ids_by_query=retrieved_chunks_by_query,
        relevant_ids_by_query=relevant_chunks_by_query,
        k=1,
    )

    chunk_at_5 = evaluate_retrieval(
        retrieved_ids_by_query=retrieved_chunks_by_query,
        relevant_ids_by_query=relevant_chunks_by_query,
        k=5,
    )

    doc_at_1 = evaluate_retrieval(
        retrieved_ids_by_query=retrieved_docs_by_query,
        relevant_ids_by_query=relevant_docs_by_query,
        k=1,
    )

    doc_at_5 = evaluate_retrieval(
        retrieved_ids_by_query=retrieved_docs_by_query,
        relevant_ids_by_query=relevant_docs_by_query,
        k=5,
    )

    return (
        {
            "chunk_hit_1": chunk_at_1.hit_rate_at_k,
            "chunk_mrr_5": chunk_at_5.mrr_at_k,
            "doc_hit_1": doc_at_1.hit_rate_at_k,
            "doc_mrr_5": doc_at_5.mrr_at_k,
        },
        top_docs_by_query,
        retrieved_docs_by_query,
    )


def evaluate_complementarity(
    benchmark_queries: list[dict],
    bm25_top_docs: list[str | None],
    dense_top_docs: list[str | None],
    bm25_ranked_docs: list[list[str]],
    dense_ranked_docs: list[list[str]],
) -> dict[str, int | float]:

    both_correct = 0
    bm25_only = 0
    dense_only = 0
    both_wrong = 0

    both_hit_5 = 0
    bm25_only_hit_5 = 0
    dense_only_hit_5 = 0
    neither_hit_5 = 0

    top_1_agreement = 0

    for (
        benchmark_query,
        bm25_doc_id,
        dense_doc_id,
        bm25_docs,
        dense_docs,
    ) in zip(
        benchmark_queries,
        bm25_top_docs,
        dense_top_docs,
        bm25_ranked_docs,
        dense_ranked_docs,
        strict=True,
    ):
        relevant_doc_ids = benchmark_query["relevant_doc_ids"]

        bm25_correct = bm25_doc_id is not None and bm25_doc_id in relevant_doc_ids

        dense_correct = dense_doc_id is not None and dense_doc_id in relevant_doc_ids

        if bm25_correct and dense_correct:
            both_correct += 1
        elif bm25_correct:
            bm25_only += 1
        elif dense_correct:
            dense_only += 1
        else:
            both_wrong += 1

        bm25_hit_5 = bool(set(bm25_docs[:5]) & relevant_doc_ids)

        dense_hit_5 = bool(set(dense_docs[:5]) & relevant_doc_ids)

        if bm25_hit_5 and dense_hit_5:
            both_hit_5 += 1
        elif bm25_hit_5:
            bm25_only_hit_5 += 1
        elif dense_hit_5:
            dense_only_hit_5 += 1
        else:
            neither_hit_5 += 1

        if bm25_doc_id is not None and bm25_doc_id == dense_doc_id:
            top_1_agreement += 1

    total = len(benchmark_queries)

    oracle_hits_1 = both_correct + bm25_only + dense_only

    union_hits_5 = both_hit_5 + bm25_only_hit_5 + dense_only_hit_5

    return {
        "both_correct": both_correct,
        "bm25_only": bm25_only,
        "dense_only": dense_only,
        "both_wrong": both_wrong,
        "oracle_hit_1": oracle_hits_1 / total,
        "top_1_agreement": top_1_agreement / total,
        "both_hit_5": both_hit_5,
        "bm25_only_hit_5": bm25_only_hit_5,
        "dense_only_hit_5": dense_only_hit_5,
        "neither_hit_5": neither_hit_5,
        "union_hit_5": union_hits_5 / total,
    }


def main() -> None:
    benchmark_queries = resolve_ground_truth(load_benchmark_queries())

    bm25, dense, hybrid = RetrievalPipeline(
        config_filepath=BENCHMARK_CONFIG_FILE_PATH,
    ).build()

    retrieval_service = RetrievalService(
        retrievers={
            "bm25": bm25,
            "dense": dense,
            "hybrid": hybrid,
        }
    )

    benchmark_results = {}
    top_docs_by_mode = {}
    ranked_docs_by_mode = {}

    for mode in ("bm25", "dense", "hybrid"):
        metrics, top_docs, ranked_docs = evaluate_mode(
            retrieval_service,
            mode,
            benchmark_queries,
        )

        benchmark_results[mode] = metrics
        top_docs_by_mode[mode] = top_docs
        ranked_docs_by_mode[mode] = ranked_docs

    complementarity = evaluate_complementarity(
        benchmark_queries=benchmark_queries,
        bm25_top_docs=top_docs_by_mode["bm25"],
        dense_top_docs=top_docs_by_mode["dense"],
        bm25_ranked_docs=ranked_docs_by_mode["bm25"],
        dense_ranked_docs=ranked_docs_by_mode["dense"],
    )

    print("\n")
    print("SIGNALRANK RETRIEVAL BENCHMARK")
    print("=" * 72)

    print(
        f"{'Mode':<10}"
        f"{'Chunk Hit@1':>15}"
        f"{'Chunk MRR@5':>15}"
        f"{'Doc Hit@1':>15}"
        f"{'Doc MRR@5':>15}"
    )

    print("-" * 72)

    for mode, metrics in benchmark_results.items():
        print(
            f"{mode.upper():<10}"
            f"{metrics['chunk_hit_1']:>15.3f}"
            f"{metrics['chunk_mrr_5']:>15.3f}"
            f"{metrics['doc_hit_1']:>15.3f}"
            f"{metrics['doc_mrr_5']:>15.3f}"
        )

    print("\n")
    print("BM25 × DENSE COMPLEMENTARITY")
    print("=" * 40)

    print(f"{'Both correct':<24}{complementarity['both_correct']:>8}")

    print(f"{'BM25 only':<24}{complementarity['bm25_only']:>8}")

    print(f"{'Dense only':<24}{complementarity['dense_only']:>8}")

    print(f"{'Both wrong':<24}{complementarity['both_wrong']:>8}")

    print("-" * 40)

    print(f"{'Oracle Hit@1':<24}{complementarity['oracle_hit_1']:>8.3f}")

    print(f"{'Top-1 agreement':<24}{complementarity['top_1_agreement']:>8.3f}")

    print("\n")
    print("BM25 × DENSE TOP-5 COMPLEMENTARITY")
    print("=" * 40)

    print(f"{'Both hit @5':<24}{complementarity['both_hit_5']:>8}")

    print(f"{'BM25 only @5':<24}{complementarity['bm25_only_hit_5']:>8}")

    print(f"{'Dense only @5':<24}{complementarity['dense_only_hit_5']:>8}")

    print(f"{'Neither @5':<24}{complementarity['neither_hit_5']:>8}")

    print("-" * 40)

    print(f"{'Union Hit@5':<24}{complementarity['union_hit_5']:>8.3f}")


if __name__ == "__main__":
    main()

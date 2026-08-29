import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import logfire
from qdrant_client import QdrantClient, models

from signalrank.config.settings import QdrantConfig


def create_qdrant_client(
    *,
    url: str | None = None,
    api_key: str | None = None,
    path: Path | None = None,
) -> QdrantClient:
    if url is not None:
        return QdrantClient(
            url=url,
            api_key=api_key,
        )

    if path is not None:
        return QdrantClient(
            path=str(path),
        )

    return QdrantClient(":memory:")


def create_configured_qdrant_client(
    config: QdrantConfig,
) -> QdrantClient:
    if config.mode == "remote":
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url:
            raise RuntimeError("QDRANT_URL is required when Qdrant mode is 'remote'")

        if not api_key:
            raise RuntimeError(
                "QDRANT_API_KEY is required when Qdrant mode is 'remote'"
            )

        return create_qdrant_client(
            url=url,
            api_key=api_key,
        )

    if config.mode == "local":
        if config.path is None:
            raise RuntimeError("Qdrant path is required when mode is 'local'")

        return create_qdrant_client(
            path=config.path,
        )

    return create_qdrant_client()


class QdrantVectorStore:
    def __init__(
        self,
        dimension: int,
        collection_name: str = "signalrank",
        client: QdrantClient | None = None,
    ):
        self._collection_name = collection_name
        self._dimension = dimension

        self._client = client or QdrantClient(":memory:")

        with logfire.span(
            "Initialize Qdrant vector store",
            collection=collection_name,
            dimension=dimension,
        ):
            collection_exists = self._client.collection_exists(
                collection_name,
            )

            if not collection_exists:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE,
                    ),
                )

                logfire.info(
                    "Qdrant collection created",
                    collection=collection_name,
                    dimension=dimension,
                )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(
            uuid5(
                NAMESPACE_URL,
                f"signalrank:{chunk_id}",
            )
        )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, object]],
    ) -> None:

        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids, vectors, and payloads must have the same length")

        points = []

        for chunk_id, vector, payload in zip(
            ids,
            vectors,
            payloads,
        ):
            if len(vector) != self._dimension:
                raise ValueError(
                    f"Vector dimension mismatch for "
                    f"{chunk_id}: expected "
                    f"{self._dimension}, got "
                    f"{len(vector)}"
                )

            point_payload = dict(payload)
            point_payload["chunk_id"] = chunk_id

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk_id),
                    vector=vector,
                    payload=point_payload,
                )
            )

        with logfire.span(
            "Qdrant upsert",
            collection=self._collection_name,
            point_count=len(points),
            dimension=self._dimension,
        ):
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:

        if len(query_vector) != self._dimension:
            raise ValueError(
                "Query vector dimension mismatch: "
                f"expected {self._dimension}, "
                f"got {len(query_vector)}"
            )

        with logfire.span(
            "Qdrant search",
            collection=self._collection_name,
            top_k=top_k,
            dimension=self._dimension,
        ):
            result = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )

        results: list[tuple[str, float]] = []

        for point in result.points:
            payload = point.payload

            if payload is None:
                raise RuntimeError("Qdrant search result is missing its payload.")

            chunk_id = payload.get("chunk_id")

            if chunk_id is None:
                raise RuntimeError("Qdrant search result is missing chunk_id.")

            results.append(
                (
                    str(chunk_id),
                    float(point.score),
                )
            )

        return results

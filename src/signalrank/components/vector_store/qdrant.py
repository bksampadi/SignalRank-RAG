from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models


class QdrantVectorStore:
    def __init__(
            self,
            dimension: int,
            collection_name: str = "signalrank",
            client: QdrantClient | None = None
    ):
        self._collection_name = collection_name

        self._client = client or QdrantClient(":memory:")

        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
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
        points = []

        for chunk_id, vector, payload in zip(
            ids,
            vectors,
            payloads,
        ):
            point_payload = dict(payload)
            point_payload["chunk_id"] = chunk_id

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk_id),
                    vector=vector,
                    payload=point_payload,
                )
            )


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
        result = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            (
                str(point.payload["chunk_id"]), 
                float(point.score),
            )
            for point in result.points
        ]
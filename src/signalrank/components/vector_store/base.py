from typing import Protocol

class VectorStore(Protocol):
    def upsert(
            self,
            ids: list[str],
            vectors: list[list[float]],
            payloads: list[dict[str, object]],
    ) -> None:
        ...

    def search(
            self,
            query_vector: list[float],
            top_k: int = 10,
    ) -> list[tuple[str, float]]:
        ...
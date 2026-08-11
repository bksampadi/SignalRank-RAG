from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def embed_documents(
            self,
            text: list[str],
    ) -> list[list[float]]:
        ...

    def embed_query(
            self,
            text: str,
    ) -> list[float]:
        ...
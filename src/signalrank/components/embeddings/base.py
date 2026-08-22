from typing import Literal, Protocol

EmbeddingProviderType = Literal[
    "sentence_transformer",
    "gemini",
]


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]: ...

    def embed_query(
        self,
        text: str,
    ) -> list[float]: ...

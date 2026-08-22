from signalrank.components.embeddings.base import (
    EmbeddingProvider,
    EmbeddingProviderType,
)
from signalrank.components.embeddings.gemini import GeminiEmbedding
from signalrank.components.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)


def create_embedding_provider(
    provider: EmbeddingProviderType,
    *,
    model_name: str | None = None,
    api_key: str | None = None,
    dimension: int | None = None,
) -> EmbeddingProvider:

    if provider == "sentence_transformer":
        return SentenceTransformerEmbedding(
            model_name=(model_name or "sentence-transformers/all-mpnet-base-v2")
        )

    if provider == "gemini":
        return GeminiEmbedding(
            model_name=model_name or "gemini-embedding-2",
            dimension=dimension if dimension is not None else 768,
            api_key=api_key,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")

from signalrank.components.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)


def test_embedding_dimension():
    provider = SentenceTransformerEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    assert provider.dimension == 768


def test_embed_query_returns_expected_dimension():
    provider = SentenceTransformerEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    embedding = provider.embed_query("RAG application")

    assert len(embedding) == provider.dimension


def test_embed_documents_preserves_counts():
    provider = SentenceTransformerEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    embeddings = provider.embed_documents(["first chunk", "second chunk"])

    assert len(embeddings) == 2
    assert all(len(embedding) == provider.dimension for embedding in embeddings)

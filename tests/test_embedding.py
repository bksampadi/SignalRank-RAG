import pytest

from signalrank.components.embeddings.sentence_transformer import SentenceTransformer, SentenceTransformerEmbedding

def test_embedding_dimension():
    provider = SentenceTransformerEmbedding()

    assert provider.dimension == 384


def test_embed_query_returns_expected_dimension():
    provider = SentenceTransformerEmbedding()

    embedding = provider.embed_query("RAG application")

    assert len(embedding) == provider.dimension


def test_embed_documents_preserves_counts():
    provider = SentenceTransformerEmbedding()

    embeddings = provider.embed_documents(
        ["first chunk", "second chunk"]
    )

    assert len(embeddings) == 2

import logfire
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedding:
    def __init__(
        self,
        model_name: str = ("sentence-transformers/all-mpnet-base-v2"),
    ):
        self.model = SentenceTransformer(model_name)

        logfire.info(
            "SentenceTransformer embeddings initialized",
            model=model_name,
            dimension=self.dimension,
        )

    @property
    def dimension(self) -> int:
        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError("Embedding dimension could not be determined.")

        return dimension

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        with logfire.span(
            "Embed documents",
            provider="sentence_transformer",
            document_count=len(texts),
            dimension=self.dimension,
        ):
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
            )

        return embeddings.tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        with logfire.span(
            "Embed query",
            provider="sentence_transformer",
            dimension=self.dimension,
        ):
            embedding = self.model.encode(text, normalize_embeddings=True)

        return embedding.tolist()

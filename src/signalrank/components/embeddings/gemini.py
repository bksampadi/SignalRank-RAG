import logfire

from langchain_google_genai import GoogleGenerativeAIEmbeddings


class GeminiEmbedding:
    def __init__(
            self,
            model_name: str,
            dimension: int,
            batch_size: int = 50,
            api_key: str | None = None,
    ):
        if batch_size <=0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        self._dimension = dimension
        self._batch_size = batch_size

        self.model = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
            output_dimensionality=dimension,
        )

        logfire.info(
            "Gemini embeddings initialized",
            model=model_name,
            dimension=dimension,
            batch_size=batch_size,
        )


    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(
            self,
            texts: list[str],
    ) -> list[list[float]]:

        with logfire.span(
            "Embed documents",
            provider="gemini",
            document_count=len(texts),
            dimension=self.dimension,
            batch_size=self._batch_size,
        ):
            return self.model.embed_documents(
                texts,
                batch_size=self._batch_size,
            )

    def embed_query(
            self,
            text: str,
    ) -> list[float]:

        with logfire.span(
            "Embed query",
            provider="gemini",
            dimension=self.dimension,
        ):
            return self.model.embed_query(text)
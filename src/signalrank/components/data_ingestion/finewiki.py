from collections.abc import Iterator

from datasets import load_dataset

from signalrank.components.data_ingestion.document import (
    DocumentElement,
    ParsedDocument,
)


class FineWikiSource:
    """
    Stream FineWiki articles as SignalRannk documents.
    """

    def __init__(
        self,
        language: str = "en",
    ):

        self._language = language

    def stream_documents(
        self,
        limit: int | None = None,
    ) -> Iterator[ParsedDocument]:
        dataset = load_dataset(
            "HuggingFaceFW/finewiki",
            name=self._language,
            split="train",
            streaming=True,
        )

        for index, row in enumerate(dataset):
            if limit is not None and index >= limit:
                break

            text = row["text"].strip()

            if not text:
                continue

            yield ParsedDocument(
                doc_id=row["id"],
                source_path=row["url"],
                file_type=".md",
                elements=(
                    DocumentElement(
                        text=text,
                        element_type="article",
                        element_index=0,
                        metadata={
                            "title": row["title"],
                            "url": row["url"],
                            "wikidata_id": row["wikidata_id"],
                            "date_modified": row["date_modified"],
                            "language": row["in_language"],
                        },
                    ),
                ),
                metadata={
                    "title": row["title"],
                    "url": row["url"],
                    "source": "FineWiki",
                    "license": "CC BY-SA 4.0",
                },
            )

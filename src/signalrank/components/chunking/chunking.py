from __future__ import annotations

import hashlib

import logfire

from signalrank.components.chunking.chunk import DocumentChunk
from signalrank.components.data_ingestion.document import ParsedDocument
from signalrank.config.settings import ChunkingConfig


class DocumentChunker:
    """
    Deterministically split parsed documents into overlapping text chunks.
    """

    def __init__(
            self,
            config: ChunkingConfig
    ):
        self.config = config

    def chunk_documents(
            self,
            documents: list[ParsedDocument],
    ) -> list[DocumentChunk]:
        """
        Chunk a collection of parsed documents.
        """

        with logfire.span(
            "Document chunking",
            documents=len(documents),
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        ) as span:
            chunks: list[DocumentChunk] = []

            for document in documents:
                chunks.extend(self.chunk_document(document))

            span.set_attribute("chunk_created", len(chunks))

            return chunks


    def chunk_document(
            self,
            document: ParsedDocument,
    ) -> list[DocumentChunk]:
        """"
        Chunk one parsed document.
        """

        text, element_spans = self._flatten_document(document)

        if not text:
            return []

        chunks: list[DocumentChunk] = []

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(
                start + self.config.chunk_size,
                len(text),
            )

            chunk_text = text[start:end]

            element_indices = tuple(
                element_index
                for (
                    element_start,
                    element_end,
                    element_index,
                    _,
                ) in element_spans
                if element_end > start
                and element_start < end

            )

            element_types = tuple(
                element_type
                for (
                    element_start,
                    element_end,
                    _,
                    element_type,
                ) in element_spans
                if element_end > start
                and element_start < end
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=self._create_chunk_id(
                        document=document,
                        chunk_index=chunk_index,
                        start=start,
                        end=end,
                        text=chunk_text
                    ),
                    doc_id=document.doc_id,
                    source_path=document.source_path,
                    file_type=document.file_type,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                    element_indices=element_indices,
                    element_types=element_types,
                    metadata=dict(document.metadata),
                )
            )

            if end == len(text):
                break

            start = end - self.config.chunk_overlap
            chunk_index += 1

        return chunks

    @staticmethod
    def _flatten_document(
        document: ParsedDocument,
    ) -> tuple[
        str,
        list[tuple[int, int, int, str]],
    ]:
        """
        Flatten document elements while retaining their character spans.

        Each span contains:
        (start, end, element_index, element_type)
        """

        parts: list[str] = []
        spans: list[tuple[int, int, int, str]] = []

        cursor = 0

        for element in document.elements:
            element_text = element.text.strip()

            if not element_text:
                continue

            if parts:
                separator = "\n\n"
                parts.append(separator)
                cursor += len(separator)

            start = cursor

            parts.append(element_text)
            cursor += len(element_text)

            spans.append(
                (
                    start,
                    cursor,
                    element.element_index,
                    element.element_type,
                )
            )

        return "".join(parts), spans
    
    @staticmethod
    def _create_chunk_id(
        document: ParsedDocument,
        chunk_index: int,
        start: int,
        end: int,
        text: str,
    ) -> str:
        """
        Create a reproducible content-sensitive chunk ID.
        """

        identity = (
            f"{document.doc_id}\0"
            f"{chunk_index}\0"
            f"{start}\0"
            f"{end}\0"
            f"{text}"
        )

        digest = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:16]

        return f"chunk_{digest}"
            
from pathlib import Path

import logfire
import pymupdf

from signalrank.components.data_ingestion.document import DocumentElement


def load_pdf(file_path: Path) -> list[DocumentElement]:
    """Extract page-aware text from a PDF."""

    with logfire.span(
        "Load PDF",
        file_path=str(file_path),
    ) as span:
        try:
            pages: list[DocumentElement] = []
            pages_without_text = 0

            with pymupdf.open(file_path) as document:
                span.set_attribute(
                    "page_count",
                    len(document),
                )

                for page_index, page in enumerate(document.pages()):
                    text = page.get_text(
                        "text",
                        sort=True,
                    ).strip()

                    if not text:
                        pages_without_text += 1
                        continue

                    pages.append(
                        DocumentElement(
                            text=text,
                            element_type="page",
                            element_index=page_index,
                            metadata={
                                "page_number": page_index + 1,
                            },
                        )
                    )

            span.set_attribute(
                "pages_extracted",
                len(pages),
            )
            span.set_attribute(
                "pages_without_text_skipped",
                pages_without_text,
            )

            return pages

        except Exception:
            logfire.exception(
                "PDF loading failed",
                file_path=str(file_path),
            )
            raise

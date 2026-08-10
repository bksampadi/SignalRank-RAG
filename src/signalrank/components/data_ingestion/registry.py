from __future__ import annotations


from collections.abc import Callable
from functools import partial
from pathlib import Path


from signalrank.components.data_ingestion.document import DocumentElement
from signalrank.components.data_ingestion.loaders.html import load_html
from signalrank.components.data_ingestion.loaders.pdf import load_pdf
from signalrank.components.data_ingestion.loaders.text import load_text


Loader = Callable[[Path], list[DocumentElement]]


def get_loader(
        extension: str,
        *,
        encoding: str = "utf-8",
) -> Loader:
    """
    Return the appropriate loader for a file extension.
    """

    extension = extension.lower()

    if extension == ".pdf":
        return load_pdf

    if extension in {".txt", ".md", ".markdown"}:
        return partial(
            load_text,
            encoding=encoding,
        )

    if extension in {".html", ".htm"}:
        return partial(
            load_html,
            encoding=encoding,
        )
    
    raise ValueError(
        f"No loader registered for extension: {extension}"
    )
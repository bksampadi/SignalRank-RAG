import hashlib
from pathlib import (
    Path,
    PurePosixPath,
)

def read_text_file(
        file_path: Path,
        encoding: str = "utf-8",
) -> str:
    return file_path.read_text(
        encoding=encoding,
    )

def create_document_id(
        source_reference: str,
        text: str,
        ) -> str:
    """
    Create a reproducible document ID from its relative source path
    and normalised content
    """
    normalised_text = text.replace("\r\n", "\n").replace("\r", "\n")
    identity = f"{source_reference}\0{normalised_text}"

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    return f"doc_{digest}"

def get_file_metadata(
        source_reference: str,
        text: str,
        ) -> dict[str, object]:

    source = PurePosixPath(source_reference)

    return {
        "source": source.as_posix(),
        "file_name": source.name,
        "file_extension": source.suffix.lower(),
        "char_count": len(text),
        "word_count": len(text.split()),
    }
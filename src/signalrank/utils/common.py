import hashlib
from pathlib import Path

def read_text_file(
        file_path: Path,
        encoding: str = "utf-8",
) -> str:
    return file_path.read_text(
        encoding=encoding,
        errors="replace",
    )

def create_document_id(file_path: Path) -> str:
    resolved_path = str(file_path.resolve()).replace("\\", "/")
    digest = hashlib.sha256(
        resolved_path.encode("utf-8")
    ).hexdigest()[:16]

    return f"{file_path.stem}-{digest}"

def get_file_metadata(file_path: Path, text:str) -> dict[str, object]:
    return {
        "source": str(file_path),
        "file_name": file_path.name,
        "file_extension": file_path.suffix.lower(),
        "char_count": len(text),
        "word_count": len(text.split()),
    }
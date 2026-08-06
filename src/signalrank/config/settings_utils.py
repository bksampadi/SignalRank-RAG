def _normalise_extension(extension: str) -> str:
    """Normalize an extension to lowercase with a leading dot."""
    extension = extension.strip().lower()

    if not extension:
        raise ValueError("Supported extensions cannot contain empty values")

    if not extension.startswith("."):
        extension = f".{extension}"

    return extension

def normalise_supported_extensions(
              extensions: tuple[str, ...],
              ) -> tuple[str, ...]:
                return tuple(
                       dict.fromkeys(
                            _normalise_extension(ext)
                            for ext in extensions
                        )
                )
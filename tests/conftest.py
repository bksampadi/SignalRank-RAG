import pytest

from signalrank.observability.logfire_config import configure_observability

def pytest_configure(config: pytest.Config) -> None:
    """Configure observability before test collection and execution."""
    configure_observability()
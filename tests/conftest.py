import pytest

from signalrank.observability.logfire_config import configure_observability

@pytest.fixture(scope="session", autouse=True)
def configure_logfire() -> None:
    configure_observability()
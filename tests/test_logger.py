from src.signalrank.logging.logger import logging
from src.signalrank.logging.logger import LOG_FILE_PATH
from pathlib import Path

def test_log_directory_exists():
    assert Path(LOG_FILE_PATH).parent.exists()


def test_logging(caplog):
    with caplog.at_level(logging.INFO):
        logging.info("Application started")

    assert "Application started" in caplog.text
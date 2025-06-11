import logging
import pytest
from ml_training_base.utils.logging.logging_utils import configure_logger

def test_configure_logger(tmp_path):
    """
    Tests the logger configuration and file output using pytest's tmp_path fixture.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory path object provided by the pytest fixture.
    """
    # 1. Create a path to a log file inside the temporary directory
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "test.log"

    # 2. Configure the logger to use this path
    logger = configure_logger(log_path=str(log_path))

    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 2

    # 3. Log test messages
    logger.info("Test INFO log message")
    logger.debug("Test DEBUG log message")

    # 4. Close all handlers to ensure buffers are flushed to disk
    #    A fresh list of handlers is given by the root logger's manager
    for handler in logging.getLogger(logger.name).handlers:
        handler.flush()
        handler.close()

    # 5. Read the file and assert its contents
    log_contents = log_path.read_text()
    assert "Test INFO log message" in log_contents
    assert "Test DEBUG log message" in log_contents

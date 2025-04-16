import os
import logging
import tempfile
import pytest
from ml_training_base.data.utils.logging_utils import configure_logger

def test_configure_logger():
    # Use a temporary file for logging.
    with tempfile.NamedTemporaryFile(delete=False) as temp_log_file:
        log_path = temp_log_file.name

    logger = configure_logger(log_path=log_path)

    # Check that we got a valid Logger object.
    assert isinstance(logger, logging.Logger)

    # We expect two handlers: console (StreamHandler) and file (FileHandler).
    handlers = logger.handlers
    assert len(handlers) == 2

    # Check presence of StreamHandler and FileHandler.
    handler_types = {type(h) for h in handlers}
    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types

    # Log a test message
    logger.info("Test INFO log message")
    logger.debug("Test DEBUG log message")

    # Cleanup: Flush and close handlers, remove temp file.
    for handler in handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()

    # Check that the log file was created and not empty.
    with open(log_path, "r") as f:
        log_contents = f.read()
        print(log_contents)
        assert "Test INFO log message" in log_contents
        assert "Test DEBUG log message" in log_contents

    os.remove(log_path)  # Remove the temporary file

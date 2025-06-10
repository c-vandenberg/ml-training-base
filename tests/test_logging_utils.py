import os
import logging
import tempfile
from ml_training_base.utils.logging.logging_utils import configure_logger

def test_configure_logger():
    """
    Tests the logger configuration and file output.
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log_file:
        log_path = temp_log_file.name

    logger = configure_logger(log_path=log_path)
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 2

    # Log test messages
    logger.info("Test INFO log message")
    logger.debug("Test DEBUG log message")

    # Cleanup: Closing the handlers is sufficient to flush the buffer to disk.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler) # Good practice to remove after closing

    # Now, read the file and assert its contents
    with open(log_path, "r") as f:
        log_contents = f.read()
        assert "Test INFO log message" in log_contents
        assert "Test DEBUG log message" in log_contents

    os.remove(log_path)

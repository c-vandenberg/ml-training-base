import logging
import pytest
from ml_training_base.utils.logging.logging_utils import configure_logger

LOGGER_NAME = "ml_training_base.data.utils.logging_utils"


@pytest.fixture
def clean_logger():
    """A pytest fixture to ensure the logger is clean before and after a test."""
    # --- Setup: Get the logger and remove any existing handlers ---
    logger = logging.getLogger(LOGGER_NAME)

    # Store original handlers to restore them later if needed, though we'll clear
    original_handlers = logger.handlers[:]

    # Remove all handlers for a clean slate
    for handler in original_handlers:
        logger.removeHandler(handler)

    # --- Yield control to the test ---
    yield logger

    # --- Teardown: Clean up after the test is done ---
    # It's good practice to remove handlers added during the test
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_configure_logger(tmp_path, clean_logger):
    """
    Tests the logger configuration and file output using a clean logger state.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory path object provided by the pytest fixture.
    clean_logger : logging.Logger
        A clean logger instance provided by our custom fixture.
    """
    # 1. Create a path to a log file inside the temporary directory
    log_path = tmp_path / "test.log"

    # 2. Configure the logger. It will now add its handlers to the clean logger.
    logger = configure_logger(log_path=str(log_path))

    # Assert that our specific logger instance was configured
    assert logger.name == LOGGER_NAME
    assert len(logger.handlers) == 2, "Expected a StreamHandler and a FileHandler"

    # 3. Log test messages
    logger.info("This is the test message.")

    # 4. Manually close the handlers to ensure the file buffer is flushed
    for handler in logger.handlers:
        handler.close()

    # 5. Read the file and assert its contents
    log_contents = log_path.read_text()
    assert "This is the test message." in log_contents

import logging
import pytest

from ml_training_base.utils.logging_utils import configure_logger

LOGGER_NAME = "ml_training_base.utils.logging_utils"


@pytest.fixture
def clean_logger():
    """
    A pytest fixture to ensure the logger is clean before and after a test.
    """
    # 1. Setup: Get the logger and remove any existing handlers
    logger = logging.getLogger(LOGGER_NAME)

    # 2. Store original handlers to restore them later if needed
    original_handlers = logger.handlers[:]

    # 3. Remove all handlers for a clean slate
    for handler in original_handlers:
        logger.removeHandler(handler)

    # 4. Yield control to the test
    yield logger

    # 5. Teardown: Clean up after the test is done
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

    # 2. Configure the logger.
    logger = configure_logger(log_path=str(log_path))

    # 3. Assert that the specific test logger instance was configured
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

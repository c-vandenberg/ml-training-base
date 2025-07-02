import logging
import pytest
from pathlib import Path

from ml_training_base.utils.logging_utils import configure_single_level_logger, configure_multi_level_logger

LOGGER_NAME_SINGLE = "ml_training_base.utils.logging_utils"


@pytest.fixture
def clean_logger():
    """
    A pytest fixture to ensure the logger is clean before and after a test.
    """
    # 1. Setup: Get the logger and remove any existing handlers
    logger = logging.getLogger(LOGGER_NAME_SINGLE)

    # 2. Remove all handlers for a clean slate
    if logger.hasHandlers():
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    # 3. Yield control to the test
    yield logger

    # 4. Teardown: Clean up after the test is done
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_configure_single_level_logger(tmp_path, clean_logger):
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
    log_path = tmp_path / "test_single.log"

    # 2. Configure the logger.
    logger = configure_single_level_logger(log_path=str(log_path))

    # 3. Assert that the specific test logger instance was configured
    assert logger.name == LOGGER_NAME_SINGLE
    assert len(logger.handlers) == 2, "Expected a StreamHandler and a FileHandler"

    # 3. Log test messages
    logger.info("This is the test message.")

    # 4. Manually close the handlers to ensure the file buffer is flushed to disk
    for handler in logger.handlers:
        handler.close()

    # 5. Read the file and assert its contents
    log_contents = log_path.read_text()
    assert "This is the test message." in log_contents


def test_configure_multi_level_logger(tmp_path: Path):
    """
    Tests that the multi-level logger creates separate files and routes
    messages correctly based on their level.
    """
    # 1. Setup: Configure the logger to use a temporary directory
    log_dir = tmp_path / "multi_level_logs"
    logger_name = "multi_level_test_logger"
    logger = configure_multi_level_logger(name=logger_name, log_dir=str(log_dir))

    # 2. Act: Log one message for each level
    debug_msg = "This is a debug message."
    info_msg = "This is an info message."
    warning_msg = "This is a warning message."
    error_msg = "This is an error message."

    logger.debug(debug_msg)
    logger.info(info_msg)
    logger.warning(warning_msg)
    logger.error(error_msg)

    # 3. Manually close all handlers to ensure buffers are flushed to disk
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    # 4. Assert: Check the contents of each log file
    # 4.1. Check debug.log
    debug_log_path = log_dir / "debug.log"
    assert debug_log_path.exists()
    debug_contents = debug_log_path.read_text()
    assert debug_msg in debug_contents
    assert info_msg not in debug_contents

    # 4.2. Check info.log
    info_log_path = log_dir / "info.log"
    assert info_log_path.exists()
    info_contents = info_log_path.read_text()
    assert info_msg in info_contents
    assert debug_msg not in info_contents
    assert warning_msg not in info_contents

    # 4.3. Check warning.log
    warning_log_path = log_dir / "warning.log"
    assert warning_log_path.exists()
    warning_contents = warning_log_path.read_text()
    assert warning_msg in warning_contents
    assert info_msg not in warning_contents

    # 4.4. Check error.log
    error_log_path = log_dir / "error.log"
    assert error_log_path.exists()
    error_contents = error_log_path.read_text()
    assert error_msg in error_contents
    assert warning_msg not in error_contents
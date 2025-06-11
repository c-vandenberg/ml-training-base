import pytest
import logging
from pathlib import Path
from unittest.mock import MagicMock, call

from ml_training_base.utils.files.files_utils import write_strings_to_file

# --- Fixtures ---

@pytest.fixture
def mock_logger() -> MagicMock:
    """
    Provides a MagicMock instance for spying on logger calls.
    """
    return MagicMock(spec=logging.Logger)

# --- Test Functions ---

def test_write_strings_to_file_creates_file_with_correct_content(
    tmp_path: Path,
    mock_logger: MagicMock
):
    """
    Tests that the function correctly creates a file and writes the provided
    list of strings to it, each on a new line.
    """
    # Arrange
    output_path = tmp_path / "output.txt"
    test_data = ["line one", "line two", "line three"]

    # Act
    write_strings_to_file(
        file_path=str(output_path),
        str_list=test_data,
        logger=mock_logger
    )

    # Assert
    # 1. Check that the file was created
    assert output_path.exists()

    # 2. Check that the file content is correct
    content = output_path.read_text()
    expected_content = "line one\nline two\nline three\n"
    assert content == expected_content

    # 3. Check that the start and end log messages were called
    mock_logger.info.assert_any_call("Starting writing lines to file...")
    mock_logger.info.assert_called_with("Writing 3 lines to file completed successfully.")


def test_write_strings_to_file_logs_progress_correctly(
    tmp_path: Path,
    mock_logger: MagicMock
):
    """
    Tests that the progress logging is triggered at the correct intervals.
    """
    # Arrange
    output_path = tmp_path / "progress_output.txt"
    # Create 7 items to test logging with an interval of 3
    test_data = [f"item {i}" for i in range(1, 8)]  # Creates a list of 7 strings

    # Act
    write_strings_to_file(
        file_path=str(output_path),
        str_list=test_data,
        logger=mock_logger,
        log_interval=3,
        content_name="items"
    )

    # Assert
    # Check all calls made to the logger's info method
    expected_calls = [
        call("Starting writing items to file..."),
        call("Written 3 / 7 items to file."),
        call("Written 6 / 7 items to file."),
        call("Writing 7 items to file completed successfully.")
    ]
    mock_logger.info.assert_has_calls(expected_calls, any_order=False)


def test_write_strings_to_file_creates_directories(
        tmp_path: Path,
        mock_logger: MagicMock
):
    """
    Tests that the function creates the parent directory if it does not exist.
    """
    # Arrange
    # Define a path with a subdirectory that doesn't exist yet
    output_dir = tmp_path / "new_directory"
    output_path = output_dir / "output.txt"
    test_data = ["some data"]

    # Pre-condition check: Ensure the directory does not exist yet
    assert not output_dir.exists()

    # Act
    write_strings_to_file(
        file_path=str(output_path),
        str_list=test_data,
        logger=mock_logger
    )

    # Assert
    # Check that both the directory and the file were created
    assert output_dir.exists()
    assert output_dir.is_dir()
    assert output_path.exists()
    assert output_path.read_text() == "some data\n"

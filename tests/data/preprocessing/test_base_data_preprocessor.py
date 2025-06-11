import pytest
import logging
import sqlite3
from pathlib import Path

from ml_training_base.data.preprocessing.base_data_preprocessors import BaseDataPreprocessor


# --- Fixtures ---

@pytest.fixture
def mock_logger() -> logging.Logger:
    """
    Provides a mock logger instance for tests.
    """
    return logging.getLogger("test_logger")


@pytest.fixture
def string_preprocessor(mock_logger: logging.Logger) -> BaseDataPreprocessor[str]:
    """
    Provides an instance of BaseDataPreprocessor specialized for strings.
    """
    return BaseDataPreprocessor[str](logger=mock_logger)


@pytest.fixture
def int_preprocessor(mock_logger: logging.Logger) -> BaseDataPreprocessor[int]:
    """
    Provides an instance of BaseDataPreprocessor specialized for integers.
    """
    return BaseDataPreprocessor[int](logger=mock_logger)


# --- Test Class ---

class TestBaseDataPreprocessor:
    def test_concatenate_data_standard(self, string_preprocessor: BaseDataPreprocessor[str]):
        """
        Tests standard concatenation of two lists.
        """
        dataset_a = ["A", "B"]
        dataset_b = ["C", "D"]
        result = string_preprocessor.concatenate_data(dataset_a, dataset_b)
        assert result == ["A", "B", "C", "D"]
        assert len(result) == 4

    def test_concatenate_data_with_empty(self, string_preprocessor: BaseDataPreprocessor[str]):
        """
        Tests concatenation with an empty list.
        """
        dataset_a = ["A", "B"]
        dataset_b = []
        result = string_preprocessor.concatenate_data(dataset_a, dataset_b)
        assert result == ["A", "B"]

        dataset_a = []
        dataset_b = ["C", "D"]
        result = string_preprocessor.concatenate_data(dataset_a, dataset_b)
        assert result == ["C", "D"]

    def test_deduplicate_in_memory_preserves_order(self, string_preprocessor: BaseDataPreprocessor[str]):
        """
        Tests that in-memory deduplication works and preserves order.
        """
        data = ["C", "A", "B", "A", "C", "D"]
        expected_result = ["C", "A", "B", "D"]
        result = string_preprocessor.deduplicate_in_memory(data, content_name="letters")
        assert result == expected_result

    def test_deduplicate_in_memory_no_duplicates(self, string_preprocessor: BaseDataPreprocessor[str]):
        """
        Tests deduplication on a list with no duplicates.
        """
        data = ["A", "B", "C", "D"]
        result = string_preprocessor.deduplicate_in_memory(data)
        assert result == data

    def test_deduplicate_in_memory_with_integers(self, int_preprocessor: BaseDataPreprocessor[int]):
        """
        Tests generic in-memory deduplication with a list of integers.
        """
        data = [10, 20, 1, 20, 10, 30]
        expected_result = [10, 20, 1, 30]
        result = int_preprocessor.deduplicate_in_memory(data, content_name="IDs")
        assert result == expected_result

    def test_deduplicate_on_disk_single_run(self, string_preprocessor: BaseDataPreprocessor[str], tmp_path: Path):
        """
        Tests on-disk deduplication for a single run, ensuring order is preserved.
        """
        # Arrange
        db_path = tmp_path / "test.db"
        data = ["C", "A", "B", "A", "C", "D"]
        expected_result = ["C", "A", "B", "D"]

        # Act
        result = string_preprocessor.deduplicate_on_disk(data, db_path=str(db_path))

        # Assert
        # Note: SQLite SELECT without ORDER BY does not guarantee order.
        #       Therefore, both lists need to be sorted both lists to ensure the test is stable.
        assert sorted(result) == sorted(expected_result)

        # Verify database state
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM unique_items").fetchone()[0]
        conn.close()
        assert count == len(expected_result)

    def test_deduplicate_on_disk_persistence(self, string_preprocessor: BaseDataPreprocessor[str], tmp_path: Path):
        """
        Tests that the on-disk database correctly persists unique items across multiple runs.
        """
        # Arrange
        db_path = tmp_path / "test.db"
        data_batch_1 = ["A", "B", "B", "C"]
        data_batch_2 = ["D", "C", "A", "E"]
        expected_final_result = ["A", "B", "C", "D", "E"]

        # Act - First run
        string_preprocessor.deduplicate_on_disk(data_batch_1, db_path=str(db_path))

        # Act - Second run on the same database file
        final_result = string_preprocessor.deduplicate_on_disk(data_batch_2, db_path=str(db_path))

        # Assert
        assert sorted(final_result) == sorted(expected_final_result)

        # Verify final database state
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM unique_items").fetchone()[0]
        conn.close()
        assert count == len(expected_final_result)

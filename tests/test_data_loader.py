import pytest

from ml_training_base.data.utils.logging_utils import configure_logger
from ml_training_base.data.base_supervised_data_loader import BaseSupervisedDataLoader

class ConcreteDataLoader(BaseSupervisedDataLoader):
    def load_data(self):
        return "Data loaded"
    def split_data(self):
        return "Data split"
    def get_dataset(self, data, training=True):
        return f"Dataset({data}, training={training})"
    def get_train_dataset(self):
        return "Train dataset"
    def get_valid_dataset(self):
        return "Validation dataset"
    def get_test_dataset(self):
        return "Test dataset"

@pytest.fixture
def mock_logger():
    # Reuse or create a logger using /dev/null on Unix to discard logs
    return configure_logger("/dev/null")

def test_base_data_loader_init(mock_logger):
    # Check for error if test_split + validation_split >= 1
    with pytest.raises(ValueError):
        _ = ConcreteDataLoader(
            x_data_file_path="x.csv",
            y_data_file_path="y.csv",
            test_split=0.5,
            validation_split=0.6,
            logger=mock_logger,
        )

    loader = ConcreteDataLoader(
        x_data_file_path="x.csv",
        y_data_file_path="y.csv",
        test_split=0.2,
        validation_split=0.2,
        logger=mock_logger,
    )

    assert loader._x_data_file_path == "x.csv"
    assert loader._y_data_file_path == "y.csv"
    assert loader._test_split == 0.2
    assert loader._validation_split == 0.2
    assert loader._train_split == 0.6

def test_base_data_loader_methods(mock_logger):
    loader = ConcreteDataLoader(
        x_data_file_path="x.csv",
        y_data_file_path="y.csv",
        test_split=0.2,
        validation_split=0.2,
        logger=mock_logger,
    )

    assert loader.load_data() == "Data loaded"
    assert loader.split_data() == "Data split"
    assert loader.get_dataset("raw_data") == "Dataset(raw_data, training=True)"
    assert loader.get_train_dataset() == "Train dataset"
    assert loader.get_valid_dataset() == "Validation dataset"
    assert loader.get_test_dataset() == "Test dataset"

    for handler in list(mock_logger.handlers):
        handler.close()
        mock_logger.removeHandler(handler)

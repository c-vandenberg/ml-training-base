import pytest

from ml_training_base.utils.logging.logging_utils import configure_logger
from ml_training_base.supervised.utils.data.base_supervised_data_loader import BaseSupervisedDataLoader

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    # Reuse or create a logger using /dev/null on Unix to discard logs
    return configure_logger("/dev/null")

# --- Test Classes and Functions ---

class ConcreteDataLoader(BaseSupervisedDataLoader):
    def setup_datasets(self):
        """
        Simulates setting up datasets and populates the attributes.
        """
        self._train_dataset = "Train Dataset Ready"
        self._valid_dataset = "Validation Dataset Ready"
        self._test_dataset = "Test Dataset Ready"


def test_base_data_loader_init(mock_logger):
    """
    Tests the constructor logic and split calculations.
    """
    # Test correct initialization
    loader = ConcreteDataLoader(test_split=0.2, validation_split=0.1, logger=mock_logger)
    assert loader._test_split == 0.2
    assert loader._validation_split == 0.1
    # Use pytest.approx for float comparison
    assert loader._train_split == pytest.approx(0.7)

    # Check for error if splits are invalid
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        ConcreteDataLoader(test_split=1.5, validation_split=0.1, logger=mock_logger)

    # Check for error if sum of splits is invalid
    with pytest.raises(ValueError, match="sum of `test_split` and `validation_split`"):
        ConcreteDataLoader(test_split=0.5, validation_split=0.6, logger=mock_logger)


def test_base_data_loader_getters(mock_logger):
    """
    Tests the dataset getter methods and their dependency on setup_datasets().
    """
    loader = ConcreteDataLoader(test_split=0.2, validation_split=0.1, logger=mock_logger)

    # Assert that getters fail before setup_datasets() is called
    with pytest.raises(RuntimeError, match="Dataset not set up"):
        loader.get_train_dataset()
    with pytest.raises(RuntimeError, match="Dataset not set up"):
        loader.get_valid_dataset()
    with pytest.raises(RuntimeError, match="Dataset not set up"):
        loader.get_test_dataset()

    # Call setup and then test the getters
    loader.setup_datasets()
    assert loader.get_train_dataset() == "Train Dataset Ready"
    assert loader.get_valid_dataset() == "Validation Dataset Ready"
    assert loader.get_test_dataset() == "Test Dataset Ready"

import os
import tempfile
import yaml
import pytest
import logging
from unittest.mock import patch

from ml_training_base.supervised.trainers.base_supervised_trainers import BaseSupervisedTrainer
from ml_training_base.supervised.environments.base_training_environments import BaseTrainingEnvironment

# A mock environment for testing purposes
class MockTrainingEnvironment(BaseTrainingEnvironment):
    def setup_environment(self, config):
        pass

    def _setup_framework_specific_environment(self, determinism_config):
        pass

# A minimal concrete trainer for testing the abstract base class
class ConcreteTrainer(BaseSupervisedTrainer):
    def _setup_data(self):
        pass
    def _setup_model(self):
        pass
    def _train(self):
        pass
    def _evaluate(self):
        pass
    def _save_model(self):
        pass

@pytest.fixture
def mock_config() -> dict:
    """Provides a minimal config dictionary."""
    return {
        "data": {"logger_path": "test.log"},
        "determinism": {}
    }


@pytest.fixture
def mock_config_file(mock_config: dict) -> str:
    """
    Creates a temporary YAML config file and yields its path.

    This fixture writes a given config dictionary to a temporary file,
    yields the file path to the test function, and then cleans up by
    deleting the file after the test is complete.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as tmp:
        yaml.dump(mock_config, tmp)
        tmp_path = tmp.name

    yield tmp_path

    os.remove(tmp_path)

@pytest.fixture
def mock_logger():
    """
    Provides a mock logger.
    """
    return logging.getLogger("test_logger")

def test_trainer_run_orchestration(mock_config_file: str, mock_logger: logging.Logger):
    """
    Tests that the run() method calls the pipeline steps in the correct order.
    We patch the methods to spy on them without actually running their logic.
    """
    with patch.object(ConcreteTrainer, '_setup_environment') as mock_setup_env, \
         patch.object(ConcreteTrainer, '_setup_data') as mock_setup_data, \
         patch.object(ConcreteTrainer, '_setup_model') as mock_setup_model, \
         patch.object(ConcreteTrainer, '_train') as mock_train, \
         patch.object(ConcreteTrainer, '_evaluate') as mock_evaluate, \
         patch.object(ConcreteTrainer, '_save_model') as mock_save:

        # Instantiate the trainer using the path from the mock_config_file fixture
        trainer = ConcreteTrainer(
            config_path=mock_config_file,
            training_env=MockTrainingEnvironment(logger=mock_logger)
        )

        # Run the pipeline
        trainer.run()

        # Assert that each method in the pipeline was called exactly once
        mock_setup_env.assert_called_once()
        mock_setup_data.assert_called_once()
        mock_setup_model.assert_called_once()
        mock_train.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_save.assert_called_once()

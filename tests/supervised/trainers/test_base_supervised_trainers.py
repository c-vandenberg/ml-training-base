import os
import pytest
import logging
import yaml
import tempfile
from unittest.mock import patch

from ml_training_base.supervised.trainers.base_supervised_trainers import BaseSupervisedTrainer
from ml_training_base.supervised.environments.base_training_environments import BaseTrainingEnvironment


class MockTrainingEnvironment(BaseTrainingEnvironment):
    """
    A mock environment that does nothing, for testing the trainer.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
    def setup_environment(self, config):
        pass

    def _setup_framework_specific_environment(self, determinism_config):
        pass


class ConcreteTrainer(BaseSupervisedTrainer):
    """
    A minimal concrete implementation to allow instantiation of the ABC.
    """
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
def mock_config(tmp_path) -> dict: # Use the tmp_path fixture from pytest
    """
    Provides a minimal config dictionary with a valid temporary log path.
    """
    # Create a subdirectory for logs inside the temporary path
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return {
        "data": {"logger_path": str(log_dir / "test.log")},
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

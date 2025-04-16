import os
import logging
import pytest
import tempfile

from ml_training_base.training.trainers.base_supervised_trainer import BaseSupervisedTrainer
from ml_training_base.training.environment.training_environment import TrainingEnvironment

@pytest.fixture
def mock_env_config():
    return {
        "env": {
            "determinism": {
                "python_seed": 0,
                "random_seed": 42,
                "numpy_seed": 42,
                "tf_seed": 42
            }
        },
        "data": {
            "logger_dir": "var/log/"
        }
    }

@pytest.fixture
def mock_config_file(mock_env_config):
    # Create a temp yaml config file
    content = """
    env:
    determinism:
        python_seed: 0
        random_seed: 42
        numpy_seed: 42
        tf_seed: 42

    data:
        logger_path: {}
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as tmp:
        # Also create a separate file for logs
        logger_path = mock_env_config.get('data').get('logger_dir')
        log_file = os.path.join(logger_path, "test_trainer.log")
        tmp.write(content.format(log_file))
        tmp_path = tmp.name

    yield tmp_path

    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if os.path.exists(log_file):
        os.remove(log_file)

# Minimal subclass for testing
class TestTrainer(BaseSupervisedTrainer):
    def _setup_model(self):
        self._logger.info("Model setup.")
    def _build_model(self):
        self._logger.info("Model built.")
    def _setup_callbacks(self):
        self._logger.info("Callbacks set up.")
    def _train(self):
        self._logger.info("Training...")
    def _save_model(self):
        self._logger.info("Model saved.")
    def _evaluate(self):
        self._logger.info("Model evaluated.")

class MockLogger(logging.Logger):
    def info(self, msg, **kwargs):
        pass

    def error(self, msg, **kwargs):
        pass

def test_trainer_run(mock_config_file):
    logger = MockLogger(name='mock_logger')
    trainer = TestTrainer(
        config_path=mock_config_file,
        training_env=TrainingEnvironment(logger=logger)
    )

    # Confirm the config loaded
    assert "env" in trainer.config
    assert "data" in trainer.config

    # Run the pipeline
    trainer.run()

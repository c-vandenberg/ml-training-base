import os
import logging
import random
import numpy as np
import pytest
import tensorflow as tf

from ml_training_base.training.environment.training_environment import TrainingEnvironment

@pytest.fixture
def mock_env_config_no_env():
    return {
        "data": {
            "logger_path": "var/log/test_logging.log"
        }
    }

@pytest.fixture
def mock_env_config_no_determinism():
    return {
        "env": {
            'test': {
                "python_seed": 0,
                "random_seed": 42,
                "numpy_seed": 42,
                "tf_seed": 42
            }
        },
        "data": {
            "logger_path": "var/log/test_logging.log"
        }
    }

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
            "logger_path": "var/log/test_logging.log"
        }
    }

class MockLogger(logging.Logger):
    def info(self, msg, **kwargs):
        pass

    def error(self, msg, **kwargs):
        pass

def test_no_env_config_error(mock_env_config_no_env):
    logger = MockLogger(name='mock_logger')
    train_env = TrainingEnvironment(logger)

    with pytest.raises(KeyError):
        train_env.setup_environment(mock_env_config_no_env)

def test_no_determinism_config_error(mock_env_config_no_determinism):
    logger = MockLogger(name='mock_logger')
    train_env = TrainingEnvironment(logger)

    with pytest.raises(KeyError):
        train_env.setup_environment(mock_env_config_no_determinism)

def test_training_environment_setup(mock_env_config):
    logger = MockLogger(name='mock_logger')
    train_env = TrainingEnvironment(logger)
    train_env.setup_environment(mock_env_config)

    # Check environment variables
    assert os.environ["PYTHONHASHSEED"] == "0"
    assert os.environ["TF_DETERMINISTIC_OPS"] == "1"

    # Confirm seeds: each random call repeated should yield consistent results
    random_nums_1 = [random.random() for _ in range(5)]
    np_random_nums_1 = np.random.rand(5)
    tf_random_nums_1 = tf.random.uniform((5,))

    # Re-seed
    train_env.setup_environment(mock_env_config)
    random_nums_2 = [random.random() for _ in range(5)]
    np_random_nums_2 = np.random.rand(5)
    tf_random_nums_2 = tf.random.uniform((5,))

    assert random_nums_1 == pytest.approx(random_nums_2)
    assert np.allclose(np_random_nums_1, np_random_nums_2)

    # For TF, need to convert tensor to a list or numpy array
    np.testing.assert_allclose(tf_random_nums_1.numpy(), tf_random_nums_2.numpy())

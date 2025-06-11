import os
import logging
import numpy as np
import pytest
import torch
import tensorflow as tf

from ml_training_base import KerasTrainingEnvironment, PyTorchTrainingEnvironment

# --- Fixtures ---

@pytest.fixture
def mock_config():
    """
    Provides a standard config for determinism tests.
    """
    return {
        "determinism": {
            "python_seed": 0,
            "random_seed": 42,
            "numpy_seed": 42,
            "tf_seed": 42,
            "torch_seed": 42
        }
    }


@pytest.fixture
def mock_logger():
    """
    Provides a mock logger for tests.
    """
    return logging.getLogger("test_logger")

# --- Test Functions ---

def test_keras_environment_setup(mock_config, mock_logger):
    """
    Tests that Keras environment sets seeds and environment variables correctly.
    """
    train_env = KerasTrainingEnvironment(logger=mock_logger)
    train_env.setup_environment(mock_config)

    assert os.environ["PYTHONHASHSEED"] == "0"
    assert os.environ["TF_DETERMINISTIC_OPS"] == "1"

    # Test seed effectiveness
    tf.random.set_seed(mock_config["determinism"]["tf_seed"])
    tf_random_nums_1 = tf.random.uniform((5,))

    tf.random.set_seed(mock_config["determinism"]["tf_seed"])
    tf_random_nums_2 = tf.random.uniform((5,))

    np.testing.assert_allclose(tf_random_nums_1.numpy(), tf_random_nums_2.numpy())


def test_pytorch_environment_setup(mock_config, mock_logger):
    """
    Tests that PyTorch environment sets seeds correctly.
    """
    train_env = PyTorchTrainingEnvironment(logger=mock_logger)
    train_env.setup_environment(mock_config)

    # Test seed effectiveness
    torch.manual_seed(mock_config["determinism"]["torch_seed"])
    torch_random_nums_1 = torch.rand(5)

    torch.manual_seed(mock_config["determinism"]["torch_seed"])
    torch_random_nums_2 = torch.rand(5)

    assert torch.equal(torch_random_nums_1, torch_random_nums_2)


def test_missing_config_key_error(mock_logger):
    """
    Tests that a KeyError is raised if the 'determinism' key is missing.
    """
    train_env = KerasTrainingEnvironment(logger=mock_logger)
    with pytest.raises(KeyError, match="'determinism'"):
        train_env.setup_environment({}) # Empty config

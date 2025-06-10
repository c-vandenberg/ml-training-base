import os
import random
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
import tensorflow as tf
import torch

class BaseTrainingEnvironment(ABC):
    """
    Abstract base class for setting up a training environment.

    Handles framework-agnostic setup (Python, NumPy seeds) and defines
    an abstract method for framework-specific configurations.

    Attributes
    ----------
    _logger : logging.Logger
        Logger instance for logging messages.
    """
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def setup_environment(self, config: Dict[str, Any]) -> None:
        """
        Set up the environment for deterministic training.

        This template method performs framework-agnostic setup and then calls
        a subclass-specific method to handle framework-dependent settings.

        By configuring the training environment to be deterministic, this method enables reproducibility
        across runs. It follows best practices for setting up deterministic environments, including setting the
        hash seed, configuring random number generators, configure specific frameworks (e.g. PyTorch, TensorFlow)
        for deterministic operations.

        More information on deterministic training and reproducibility can be found at:
        - NVIDIA Clara Train documentation:
            https://docs.nvidia.com/clara/clara-train-archive/3.1/nvmidl/additional_features/determinism.html
        - NVIDIA Reproducibility Framework GitHub:
            https://github.com/NVIDIA/framework-reproducibility/tree/master/doc/d9m

        Returns
        -------
        None

        Parameters
        ----------
        config : Dict[str, Any]
            A dictionary containing the configuration, expected to have a
            'determinism' key.

        Raises
        ------
        KeyError
            If the 'determinism' configuration is missing.

        Notes
        -----
        1. Sets the `PYTHONHASHSEED` environment variable to control the hash seed used by Python.
        2. Seeds Python's and NumPy's random number generators.
        """
        self._logger.info("Performing framework-agnostic environment setup...")
        determinism_config = config.get('determinism', {})

        if not determinism_config:
            raise KeyError("Configuration must have a 'determinism' section.")

        # Framework-agnostic setup
        os.environ['PYTHONHASHSEED'] = str(determinism_config.get('python_seed', 0))
        random.seed(determinism_config.get('random_seed', 42))
        np.random.seed(determinism_config.get('numpy_seed', 42))

        # Call the framework-specific implementation
        self._setup_framework_specific_environment(determinism_config)

        self._logger.info("Environment setup for deterministic (reproducible) training complete.")

    @abstractmethod
    def _setup_framework_specific_environment(self, determinism_config: Dict[str, Any]) -> None:
        """
        Set up the framework-specific parts of the environment.

        Subclasses must implement this method to configure their specific
        ML framework (e.g., TensorFlow, PyTorch) for determinism.

        Parameters
        ----------
        determinism_config : Dict[str, Any]
            The subsection of the config related to determinism.
        """
        raise NotImplementedError

class KerasTrainingEnvironment(BaseTrainingEnvironment):
    """
    Sets up a deterministic environment specifically for TensorFlow/Keras.
    """
    def _setup_framework_specific_environment(self, determinism_config: Dict[str, Any]) -> None:
        """
        Configure TensorFlow for deterministic operations.

        Parameters
        ----------
        determinism_config : Dict[str, Any]
            The subsection of the config related to determinism.

        Returns
        -------
        None

        Notes
        -----
        1. Seeds TensorFlow's random number generator.
        2. Enables deterministic operations in TensorFlow by setting `TF_DETERMINISTIC_OPS=1`.
        3. Optionally disables GPU and limits TensorFlow to single-threaded execution when
           `full_determinism` is set to `True` in the config.
            - This is because modern GPUs and CPUs are designed to execute computations in parallel across many cores.
            - This parallelism is typically managed asynchronously, meaning that the order of operations or the
            availability of computing resources can vary slightly from one run to another
            - It is this asynchronous parallelism that can introduce random noise, and hence, non-deterministic
            behaviour.
            - However, configuring TensorFlow to use the CPU (`os.environ['CUDA_VISIBLE_DEVICES'] = ''`) and
            configuring Tensorflow to use single-threaded execution severely impacts performance.
        """
        self._logger.info("Performing TensorFlow-specific environment setup...")
        tf.random.set_seed(determinism_config.get('tf_seed', 42))

        tf.random.set_seed(determinism_config['tf_seed'])

        # Configure TensorFlow for deterministic operations
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        # Optional, performance-impacting settings for full determinism
        if determinism_config.get('full_determinism', False):
            self._logger.warning("Enabling full determinism for TensorFlow. This will impact performance.")

            # OpenMP is used by many numeric libraries (e.g. Eigen, MKL etc.) to parallelize loops across
            # multiple CPU threads.
            # Configure OpemMP to use only 1 thread for deterministic operations (impacts performance)
            os.environ['OMP_NUM_THREADS'] = '1'

            # Intra-op parallelism is used by TensorFlow to parallelize individual operations (e.g. matrix
            # multiplication) across multiple threads.
            # Configure intra-op parallelism to limit any single operation to a single thread (impacts performance).
            os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

            # Inter-op parallelism is used by TensorFlow to parallelize multiple operations across multiple threads.
            # Configure inter-op parallelism to limit TensorFlow to one operation at a time (impacts performance).
            os.environ['TF_NUM_INTEROP_THREADS'] = '1'

            # Disable GPU for deterministic behavior (impacts performance).
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

            # Configure TensorFlow session for single-threaded execution (heavily impacts performance)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)


class PyTorchTrainingEnvironment(BaseTrainingEnvironment):
    """
    Sets up a deterministic environment specifically for PyTorch.
    """
    def _setup_framework_specific_environment(self, determinism_config: Dict[str, Any]) -> None:
        """Configure PyTorch for deterministic operations.

        Parameters
        ----------
        determinism_config : Dict[str, Any]
            The subsection of the config related to determinism.

        Returns
        -------
        None

        Notes
        -----
        1. Seeds PyTorch's random number generators for both CPU (`torch.manual_seed`)
           and all CUDA devices (`torch.cuda.manual_seed_all`) for consistent
           weight initialization and dropout.
        2. Optionally configures the cuDNN backend for deterministic behavior when
           `full_determinism` is set to `True` in the config.
           - Modern GPUs use non-deterministic algorithms for operations like
             convolutions because they are often faster.
           - `torch.backends.cudnn.deterministic = True` forces cuDNN to use
             deterministic algorithms, which can impact performance.
           - `torch.backends.cudnn.benchmark = False` disables a feature where
             cuDNN finds the fastest algorithm for a given input size. This
             benchmarking process can be a source of non-determinism, so it
             must be disabled for full reproducibility.
           - Achieving full determinism in PyTorch on a GPU can result in a
             performance penalty.
        """
        self._logger.info("Performing PyTorch-specific environment setup...")
        torch.manual_seed(determinism_config.get('torch_seed', 42))

        # Setup for CUDA (GPU) determinism
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(determinism_config.get('torch_seed', 42))

        if determinism_config.get('full_determinism', False):
            self._logger.warning("Enabling full determinism for PyTorch. This may impact performance.")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudunn.benchmark = False
        
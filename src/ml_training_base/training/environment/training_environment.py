import os
import random
import logging
from typing import Dict, Any

import numpy as np
import tensorflow as tf

from src.ml_training_base.training.environment.base_environment import BaseEnvironment

class TrainingEnvironment(BaseEnvironment):
    """
    TrainingEnvironment

    Provides methods to set up the training environment for deterministic (reproducible) training.

    Methods
    -------
    setup_environment(config)
        Sets up the environment for deterministic training based on the provided configuration.
    """
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def setup_environment(self, config: Dict[str, Any]) -> None:
        """
        Set up the environment for deterministic (reproducible) training.

        This method ensures that the training environment is configured to be deterministic, enabling reproducibility
        across runs. It sets seeds for Python, NumPy, and TensorFlow, and configures TensorFlow for deterministic
        operations.

        The process follows best practices for setting up deterministic environments, including setting the hash seed,
        configuring random number generators, and enabling deterministic operations in TensorFlow.

        More information on deterministic training and reproducibility can be found at:
        - NVIDIA Clara Train documentation: https://docs.nvidia.com/clara/clara-train-archive/3.1/nvmidl/additional_features/determinism.html
        - NVIDIA Reproducibility Framework GitHub: https://github.com/NVIDIA/framework-reproducibility/tree/master/doc/d9m

        Returns
        -------
        None

        Notes
        -----
        1. Sets the `PYTHONHASHSEED` environment variable to control the hash seed used by Python.
        2. Seeds Python's `random` module, NumPy, and TensorFlow's random number generators for consistency.
        3. Enables deterministic operations in TensorFlow by setting `TF_DETERMINISTIC_OPS=1`.
        4. Optionally disables GPU and limits TensorFlow to single-threaded execution.
            - This is because modern GPUs and CPUs are designed to execute computations in parallel across many cores.
            - This parallelism is typically managed asynchronously, meaning that the order of operations or the
            availability of computing resources can vary slightly from one run to another
            - It is this asynchronous parallelism that can introduce random noise, and hence, non-deterministic
            behaviour.
            - However, configuring TensorFlow to use the CPU (`os.environ['CUDA_VISIBLE_DEVICES'] = ''`) and configuring
            Tensorflow to use single-threaded execution severely impacts performance.
        """
        if config.get('env') is not None:
            env_config = config.get('env', {})
            determinism_config = env_config.get('determinism', {})
        else:
            determinism_config = config.get('determinism', {})

        if not determinism_config:
            raise KeyError("Configuration YAML file must have `env: determinism` key-value pair.")

        # Set Python's built-in hash seed
        os.environ['PYTHONHASHSEED'] = str(determinism_config['python_seed'])

        # Set seeds for random number generators
        random.seed(determinism_config['random_seed'])
        np.random.seed(determinism_config['numpy_seed'])
        tf.random.set_seed(determinism_config['tf_seed'])

        ## mixed_precision.set_global_policy('float64')

        # Configure TensorFlow for deterministic operations
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        # OpenMP is used by many numeric libraries (e.g. Eigen, MKL etc.) to parallelize loops across
        # multiple CPU threads.
        # Optionally configure OpemMP to use only 1 thread for deterministic operations (impacts performance)
        ## os.environ['OMP_NUM_THREADS'] = '1'

        # Intra-op parallelism is used by TensorFlow to parallelize individual operations (e.g. matrix multiplication)
        # across multiple threads.
        # Optionally configure intra-op parallelism to limit any single operation to a single thread
        # (impacts performance).
        ## os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

        # Inter-op parallelism is used by TensorFlow to parallelize multiple operations across multiple threads.
        # Optionally configure inter-op parallelism to limit TensorFlow to one operation at a time.
        # (impacts performance).
        ## os.environ['TF_NUM_INTEROP_THREADS'] = '1'

        # Optionally disable GPU for deterministic behavior (impacts performance).
        ## os.environ['CUDA_VISIBLE_DEVICES'] = ''

        # Configure TensorFlow session for single-threaded execution (optional, heavily impacts performance)
        ## tf.config.threading.set_intra_op_parallelism_threads(1)
        ## tf.config.threading.set_inter_op_parallelism_threads(1)

        self._logger.info("Environment setup for deterministic (reproducible) training complete.")
        
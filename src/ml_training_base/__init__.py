"""
ml-training-base: A Python package providing base classes and utilities for machine learning projects
"""
from ml_training_base.data.preprocessing.base_data_preprocessors import BaseDataPreprocessor

from ml_training_base.supervised.environments.base_training_environments import (
    BaseTrainingEnvironment,
    KerasTrainingEnvironment,
    PyTorchTrainingEnvironment
)

from ml_training_base.supervised.trainers.base_supervised_trainers import (
    BaseSupervisedTrainer,
    BaseKerasSupervisedTrainer,
    BasePyTorchSupervisedTrainer
)

from ml_training_base.supervised.utils.data.base_supervised_data_loader import BaseSupervisedDataLoader

from ml_training_base.utils.logging.logging_utils import configure_logger

from ml_training_base.utils.files.files_utils import write_strings_to_file

__all__ = [
    # Public Data Preprocessing Classes
    "BaseDataPreprocessor",

    # Public Environment Classes
    "BaseTrainingEnvironment",
    "KerasTrainingEnvironment",
    "PyTorchTrainingEnvironment",

    # Public Trainer Classes
    "BaseSupervisedTrainer",
    "BaseKerasSupervisedTrainer",
    "BasePyTorchSupervisedTrainer",

    # Public Data Loader Classes
    "BaseSupervisedDataLoader",

# Public Utility Functions
    "configure_logger",
]

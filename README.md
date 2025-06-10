# Machine Learning Training Base (ml-training-base)

`ml-training-base` is a Python package providing base classes and utilities for machine learning projects. Currently, the package only supports supervised learning.

It includes:
* A configurable logging setup for both console and file outputs. 
* Base classes for data loaders (`BaseSupervisedDataLoader`).
* An environment setup class for deterministic training (`TrainingEnvironment`), ensuring reproducible runs.
* A base trainer class (`BaseSupervisedTrainer`) that outlines a typical training workflow in supervised learning.

By using these abstractions, you can quickly spin up a new ML pipeline with consistent structure and easily extend or override specific components to suit your needs.

# Table of Contents
1. [Features](https://github.com/c-vandenberg/ml-training-base?tab=readme-ov-file#features)
2. [Installation](https://github.com/c-vandenberg/ml-training-base?tab=readme-ov-file#installation)
3. [Quick Start](https://github.com/c-vandenberg/ml-training-base?tab=readme-ov-file#quick-start)
4. [Package Structure](https://github.com/c-vandenberg/ml-training-base?tab=readme-ov-file#package-structure)
5. [Configuration File](https://github.com/c-vandenberg/ml-training-base?tab=readme-ov-file#configuration-file)
6. [License](https://github.com/c-vandenberg/ml-training-base?tab=readme-ov-file#license)

## Features
* Reusable Base Classes: Standard building blocks for data loading, training, callbacks, and environment management. 
* Logging Utilities: Automatically configure logging to both console and file, with customizable logging paths. 
* Deterministic Environment Setup: Control Python, NumPy, and TensorFlow seeds for reproducible ML experiments. 
* Clear Project Structure: Easily extend or override abstract methods in your own data loaders, trainers, or environment logic.

## Installation
You can install this package locally via:
```
pip install ml-training-base
```

## Quick Start
1. **Install** the package and its dependencies.
2. **Create** a YAML configuration file (e.g. `config.yaml`) with your environment, logging, and data settings.
3. **Import** the classes in your script or Jupyter notebook:
```
import logging
from ml_training_base.data.utils.logging_utils import configure_logger
from ml_training_base.supervised.environments.base_training_environments import KerasTrainingEnvironment
from ml_training_base.supervised.trainers.base_supervised_trainers import BaseSupervisedTrainer
```
4. **Set up** your environment and trainer:
```
# For example, a custom trainer that inherits from BaseSupervisedTrainer
class MyCustomTrainer(BaseSupervisedTrainer):
    def _setup_model(self):
        # Initialize your model here, e.g., a TensorFlow/Keras or PyTorch model
        pass

    def _build_model(self):
        # Compile or build your model
        pass

    def _setup_callbacks(self):
        # Setup your training callbacks, checkpointing, etc.
        pass

    def _train(self):
        # Implement your training loop or model.fit(...) call
        pass

    def _save_model(self):
        # Save trained model to disk
        pass

    def _evaluate(self):
        # Evaluate your model on the test set
        pass

# Usage:
trainer = MyCustomTrainer(
    config_path="path/to/config.yaml",
    training_env=KerasTrainingEnvironment(logger=logging.getLogger(__name__))
)
trainer.run()
```

## Package Structure
```
ml-training-base/
├── pyproject.toml
├── src/
│   └── ml_training_base/
│       ├── __init__.py
│       ├── data/
│       │   └── utils/
│       │       ├── __init__.py
│       │       └── logging_utils.py
│       └── supervised/
│           ├── __init__.py
│           ├── environments/
│           │   ├── __init__.py
│           │   └── base_training_environments.py
│           ├── trainers/
│           │   ├── __init__.py
│           │   └── base_supervised_trainers.py
│           └── utils/
│               └── data/
│                   ├── __init__.py
│                   └── base_supervised_data_loader.py
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_environment.py
│   ├── test_logging_utils.py
│   └── test_trainer.py
├── README.md
├── LICENSE
└── pyproject.toml
```

### Key Modules
* `data/utils/logging_utils.py`:
  * Contains `configure_logger(log_path)` utility, which sets up a standardized console and file logger for use throughout the package.
* `supervised/environments/base_training_environments.py`: 
  * Defines the `BaseEnvironment` abstract class for handling environment setup.
  * Provides concrete, framework-specific implementations like `KerasTrainingEnvironment` and `PyTorchTrainingEnvironment` that manage deterministic setup (setting seeds, configuring hardware options, etc.).
* `supervised/trainers/base_supervised_trainers.py`: 
  * Contains the core training framework hierarchy.
  * `BaseSupervisedTrainer`: The framework-agnostic abstract class that defines the training pipeline (`run()`, `_setup_model()`, `_train()`, etc.).
  * `BaseKerasSupervisedTrainer` & `BasePyTorchSupervisedTrainer`: Framework-specific abstract classes that implement common boilerplate for Keras (`model.fit()`) and PyTorch (manual training loop).
* `supervised/utils/data/base_supervised_data_loader.py`: 
  * Contains the `BaseSupervisedDataLoader` abstract class. This defines the contract for creating data preparation pipelines (`setup_datasets()`, `get_train_dataset()`, etc.) that are used by the trainers.

## Configuration File
You can define your runtime settings (e.g., logger paths, environment determinism seeds, model hyperparameters) in a YAML file. 

For example:
```
# Data Configuration and Hyperparameters
data:
  x_data_path: 'data/processed/x_data'
  y_data_path: 'data/processed/y_data'
  logger_path: 'var/log/training.log'
  batch_size: 32
  test_split: 0.1
  validation_split: 0.1

# Model Configuration and Hyperparameters
model:
  attention_dim: 512
  encoder_embedding_dim: 512
  decoder_embedding_dim: 512
  units: 512
  encoder_num_layers: 2
  decoder_num_layers: 4

# Training Configuration and Hyperparameters
training:
  epochs: 100
  early_stop_patience: 5
  weight_decay: null
  dropout_rate: 0.2
  learning_rate: 1e-4

# Environment Configuration
env:
  determinism:
    python_seed: "44478977"
    random_seed: 440651
    numpy_seed: 110789
    tf_seed: 61592
```

## License
This project is licensed under the terms of the [MIT License](https://opensource.org/license/mit).
Feel free to copy, modify, and distribute per its terms.

# ml-training-base

ml-training-base is a Python library providing base classes and utilities for supervised machine learning projects. It includes:
* A configurable logging setup for both console and file outputs. 
* Base classes for data loaders (`BaseSupervisedDataLoader`).
* An environment setup class for deterministic training (`TrainingEnvironment`), ensuring reproducible runs.
* A base trainer class (`BaseSupervisedTrainer`) that outlines a typical training workflow in supervised learning.

By using these abstractions, you can quickly spin up a new ML pipeline with consistent structure and easily extend or override specific components to suit your needs.

# Table of Contents
1. Features
2. Installation
3. Quick Start
4. Package Structure
5. Configuration File
6. License

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
from ml_training_base.training.environment.environment import TrainingEnvironment
from ml_training_base.training.trainer import BaseSupervisedTrainer
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
    training_env=TrainingEnvironment(logger=logging.getLogger(__name__))
)
trainer.run()
```

## Package Structure
```
ml-training-base/
│
├── src/
│   └── ml_training_base/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── utils/
│       │       ├── __init__.py
│       │       └── logging_utils.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── environment/
│       │   │   ├── __init__.py
│       │   │   ├── base_environment.py
│       │   │   └── environment.py
│       │   ├── trainer.py
│       │   └── ...
│       └── ...
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
  * Contains `configure_logger(log_path)` which sets up console/file logging.
* `training/environment/base_environment.py`: 
  * Abstract base class `BaseEnvironment` for environment setup tasks.
* `training/environment/training_environment.py`: 
  * Implementation of `TrainingEnvironment`, enabling deterministic training (sets seeds, configures TensorFlow ops, etc.).
* `training/trainer.py`: 
  * Contains `BaseSupervisedTrainer`, an abstract class to streamline a typical training workflow (environment setup, model creation, training loop, evaluation).

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
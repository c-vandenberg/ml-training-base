import os
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union

from ml_training_base.supervised.environments.base_training_environments import BaseTrainingEnvironment
from ml_training_base.utils.logging.logging_utils import configure_logger

import tensorflow as tf
from tensorflow.keras.callbacks import (Callback, EarlyStopping, TensorBoard,
                                        ReduceLROnPlateau, ModelCheckpoint)
import torch


class BaseSupervisedTrainer(ABC):
    """
    Abstract base class for any supervised learning trainer.

    This class is framework and architecture-agnostic. It defines the essential
    structure and sequence of a supervised training pipeline and acts as the base
    layer for all supervised learning trainers.

    Attributes
    ----------
    _config : Dict[str, Any]
        Configuration dictionary loaded from a YAML file.
    _training_env : BaseTrainingEnvironment
        Class instance for setting up the training environment (e.g. seeds
        and device configs for deterministic training).
    _logger : logging.Logger
        Logger instance for logging messages.
    """
    def __init__(self, config_path: str, training_env: BaseTrainingEnvironment):
        self._config: Dict[str, Any] = self._load_config(config_path)
        self._training_env = training_env
        self._logger = self._setup_logger()

    def run(self):
        """
        Execute the standard end-to-end training pipeline.

        The pipeline consists of the following steps in order:
        1. _setup_environment
        2. _setup_data
        3. _setup_model
        4. _train
        5. _save_model
        6. _evaluate

        Raises
        ------
        Exception
            If an error occurs during any stage of the training pipeline, it
            is logged and re-raised.
        """
        try:
            self._setup_environment()
            self._setup_data()
            self._setup_model()
            self._train()
            self._save_model()
            self._evaluate()
        except Exception as e:
            self._logger.error(f"An error occurred during the training pipeline: {e}")
            raise

    @abstractmethod
    def _setup_data(self):
        """
        Loads, preprocesses, and splits the data.

        This method must be implemented by a subclass. It is responsible for
        all data preparation steps and should populate the internal attributes
        for the train, validation, and test datasets.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a concrete subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_model(self):
        """
        Instantiates the model, optimizer, and loss function.

        This method must be implemented by a subclass. It is responsible for
        all model initialization logic and should populate the `self._model`
        attribute. This includes tasks such as:
        - Parsing model hyperparameters from `self._config`.
        - Creating the model (e.g., Keras or PyTorch model).
        - Compiling the model with an optimizer, loss, and metrics if applicable.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a concrete subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _train(self):
        """
        Execute the training loop.

        This method must be implemented by a subclass. It should define the
        core training logic, such as:
        - Iterating over the training dataset for multiple epochs.
        - Monitoring training metrics and losses.
        - Optionally validating the model on a validation dataset each epoch.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a concrete subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _save_model(self):
        """
        Save the trained model to disk.

        This method must be implemented by a subclass. It should handle:
        - Saving the full model in a framework-specific format (e.g.,
          TensorFlow SavedModel, PyTorch checkpoint).
        - Exporting to alternative formats (e.g., ONNX).
        - Saving additional artifacts like tokenizers, config files, etc.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a concrete subclass.
        """
        raise NotImplementedError

    def _evaluate(self):
        """
        Evaluate the trained model on a test dataset.

        This method must be implemented by a subclass. It should
        - Loading or preparing the test dataset.
        - Running inference to compute loss, accuracy, or other metrics.
        - Logging or storing evaluation results.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a concrete subclass.
        """
        raise NotImplementedError

    def _setup_environment(self):
        """
        Configure the training environment.

        This method uses the injected `BaseEnvironment` instance to set up
        the environment, which typically involves setting random seeds and
        configuring hardware usage (GPU/CPU) for deterministic training.
        """
        self._logger.info("Setting up training environment...")
        self._training_env.setup_environment(self._config)

    def _setup_logger(self):
        """
        Configure and return a logger instance.

        Creates the log directory if it doesn't exist and sets up a logger
        based on the path specified in the configuration file.

        Returns
        -------
        logging.Logger
            The configured logger instance.
        """
        log_path = self._config.get('data', {}).get('logger_path', 'var/log/default_logs.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        return configure_logger(log_path)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """
        Loads configuration from a YAML file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist at `config_path`.
        yaml.YAMLError
            If there is an error parsing the YAML file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, 'r') as file:
            try:
                config: Dict[str, Any] = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file: {e}")

        return config

    @property
    def config(self) -> Dict[str, Any]:
        """
        Access the loaded configuration dictionary.

        Returns
        -------
        Dict[str, Any]
            The configuration dictionary.
        """
        return self._config

class BaseKerasSupervisedTrainer(BaseSupervisedTrainer, ABC):
    """
    Abstract base class for a standard TensorFlow/Keras supervised learning workflow.

    This class serves as an intermediate layer for all TensorFLow/Keras-based trainers.
    It is framework-specific (TensorFLow/Keras) but remains architecture-agnostic.

    It provides default implementations for training environment setup, setting up common
    callbacks, model saving, and evaluation, while delegating model and data-specific
    logic to subclasses.

    Attributes
    ----------
    _model : tf.keras.Model
        The Keras model to be trained.
    _train_dataset : tf.data.Dataset
        The dataset for training.
    _valid_dataset : tf.data.Dataset
        The dataset for validation.
    _test_dataset : tf.data.Dataset
        The dataset for testing.
    _callbacks : List[tf.keras.callbacks.Callback]
        A list of callbacks to use during training.
    """
    def __init__(self, config_path: str, training_env: BaseTrainingEnvironment):
        super().__init__(config_path=config_path, training_env=training_env)
        self._model: Union[tf.keras.Model, None] = None
        self._train_dataset: Union[tf.data.Dataset, None] = None
        self._valid_dataset: Union[tf.data.Dataset, None] = None
        self._test_dataset: Union[tf.data.Dataset, None] = None
        self._callbacks: List[Callback] = []

    def run(self):
        """
        Execute the standard end-to-end Keras model training pipeline.

        The pipeline consists of the following steps in order:
        1. _setup_environment
        2. _setup_data
        3. _setup_model
        4. _build_model
        5. _setup_callbacks
        6. _train
        7. _evaluate
        8. _save_model

        Raises
        ------
        Exception
            If an error occurs during any stage of the training pipeline, it
            is logged and re-raised.
        """
        try:
            self._setup_environment()
            self._setup_data()
            self._setup_model()
            self._build_model()
            self._setup_callbacks()
            self._train()
            self._evaluate()
            self._save_model()
        except Exception as e:
            self._logger.error(f"A critical error occurred during the Keras training pipeline: {e}")
            raise

    def _build_model(self):
        """
        Build the model by running a forward pass to initialize weights.

        This method takes a single batch from the training dataset and runs a
        forward pass to trigger the lazy initialization of the model's layers.
        It then logs a summary of the built model.

        Raises
        ------
        RuntimeError
            If the training dataset has not been initialized before calling
            this method.
        """
        if not self._train_dataset:
            raise RuntimeError("The training dataset must be set up before building the model.")

        self._logger.info("Building Keras model by running a single forward pass...")

        sample_batch = next(iter(self._train_dataset))

        if isinstance(sample_batch, (list, tuple)):
            _ = self._model(sample_batch[0])
        else:
            _ = self._model(sample_batch)

        self._logger.info("Model built successfully. Logging summary:")
        self._model.summary(print_fn=self._logger.info)

    def _setup_callbacks(self):
        """
        Set up common Keras callbacks.

        This method configures and appends the following
        `tensorflow.keras.callbacks` callbacks to `self._callbacks`:
        - TensorBoard: For logging metrics to TensorBoard.
        - EarlyStopping: To stop training when a monitored metric has stopped
          improving.
        - ReduceLROnPlateau: To reduce the learning rate when a metric has
          stopped improving.
        - ModelCheckpoint: To save the best model during training.

        This method can be extended or overridden by subclasses to add custom
        callbacks.
        """
        self._logger.info("Setting up Keras callbacks...")
        train_conf = self._config.get('training', {})

        # TensorBoard
        tensorboard_dir = train_conf.get('tensorboard_dir', './tensorboard')
        self._callbacks.append(TensorBoard(log_dir=tensorboard_dir))

        # Early Stopping
        self._callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=train_conf.get('patience', 10),
            restore_best_weights=True
        ))

        # Reduce Learning Rate on Plateau
        self._callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=train_conf.get('lr_factor', 0.2),
            patience=train_conf.get('lr_patience', 5)
        ))

        # Basic Model Checkpointing
        checkpoint_dir = train_conf.get('checkpoint_dir', './checkpoints')
        self._callbacks.append(ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss'
        ))

    def _train(self):
        """
        Execute basic training using model.fit().

        This method provides a default implementation for training a Keras
        model. It can be overridden in a subclass to implement a custom
        training loop if needed.
        """
        self._logger.info("Starting Keras model training with model.fit()...")
        train_conf = self._config.get('training', {})
        self._model.fit(
            self._train_dataset,
            epochs=train_conf.get('epochs', 10),
            validation_data=self._valid_dataset,
            callbacks=self._callbacks
        )

    def _evaluate(self):
        """
        Perform a basic evaluation using `model.evaluate()`

        This method provides a default implementation for evaluating the model
        on the test set. It can be extended in a subclass to compute and log
        more detailed or custom metrics.
        """
        self._logger.info("Evaluating Keras model with model.evaluate()...")
        results = self._model.evaluate(self._test_dataset, return_dict=True)
        self._logger.info(f"Test Evaluation Results: {results}")

    def _save_model(self):
        """
        Saves the model in the standard Keras format.

        This method provides a default implementation for saving the final
        trained model. It can be extended in a subclass to save the model in
        additional formats like ONNX or HDF5.
        """
        self._logger.info("Saving Keras model...")
        model_save_dir = self._config.get('training', {}).get('model_save_dir', './model')
        save_path = os.path.join(model_save_dir, 'model.keras')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._model.save(save_path)
        self._logger.info(f"Model saved to {save_path}")

class BasePyTorchSupervisedTrainer(BaseSupervisedTrainer, ABC):
    """
    Abstract base class for a standard PyTorch supervised learning workflow.

    This class serves as an intermediate layer for all PyTorch-based trainers.
    It is framework-specific (PyTorch) but remains architecture-agnostic.

    Common PyTorch-related logic can be factored into this class later as
    patterns emerge from concrete implementations.

    Attributes
    ----------
    _model : torch.nn.Module
        The PyTorch model to be trained.
    _optimizer : torch.optim.Optimizer
        The optimizer for training the model.
    _loss_fn : Any
        The loss function.
    _device : str
        The device to run the training on ('cuda' or 'cpu').
    _train_loader : torch.utils.data.DataLoader
        The data loader for the training dataset.
    _valid_loader : torch.utils.data.DataLoader
        The data loader for the validation dataset.
    _test_loader : torch.utils.data.DataLoader
        The data loader for the test dataset.

    """
    def __init__(self, config_path: str, training_env: BaseTrainingEnvironment):
        """
        Initialise the BasePyTorchSupervisedTrainer.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        training_env : BaseEnvironment
            Class responsible for setting up the training environment.

        """
        super().__init__(config_path=config_path, training_env=training_env)

        self._model: Union[torch.nn.Module, None] = None
        self._optimizer: Union[torch.optim.Optimizer, None] = None
        self._loss_fn: Any = None
        self._device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._train_loader: Union[torch.utils.data.DataLoader, None] = None
        self._valid_loader: Union[torch.utils.data.DataLoader, None] = None
        self._test_loader: Union[torch.utils.data.DataLoader, None] = None

    pass
import os
import yaml
from typing import Dict, Any

from src.ml_training_base.training.environment.base_environment import BaseEnvironment
from src.ml_training_base.data.utils.logging_utils import configure_logger

class BaseSupervisedTrainer:
    """
    A base class for supervised learning trainers, providing a typical supervised machine
    learning training workflow from environment setup through model evaluation.

    Attributes
    ----------
    _config : Dict[str, Any]
        Configuration dictionary loaded from a YAML file.
    _training_env : BaseEnvironment
        Class or instance responsible for setting up deterministic training
        (e.g., seeds, device config).
    _logger : logging.Logger
        Logger instance used for logging messages throughout the training
        pipeline.
    """
    def __init__(
        self,
        config_path: str,
        training_env: BaseEnvironment
    ):
        """
        Initialize the BaseSupervisedTrainer with a configuration file and an
        environment setup class.

        Parameters
        ----------
        config_path : str
            Path to a YAML configuration file.
        training_env : Any, optional
            The environment class or instance to handle environment
            determinism (default is TrainingEnvironment).

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist at `config_path`.
        """
        self._config: Dict[str, Any] = self._load_config(config_path)
        self._training_env = training_env

        # Ensure log directory exists
        os.makedirs(os.path.dirname(
            self._config.get('data', {}).get('logger_path', 'var/log/default_logs.log')),
            exist_ok=True
        )

        self._logger = configure_logger(
            self._config.get('data', {}).get('logger_path', 'var/log/default_logs.log')
        )

    def run(self):
        """
        Execute the standard end-to-end training pipeline.

        The pipeline consists of the following steps in order:
            1. _setup_environment
            2. _setup_model
            3. _build_model
            4. _setup_callbacks
            5. _train
            6. _save_model
            7. _evaluate

        Raises
        ------
        Exception
            If an error occurs during any stage of the training pipeline, it
            is logged and re-raised.
        """
        try:
            self._setup_environment()
            self._setup_model()
            self._build_model()
            self._setup_callbacks()
            self._train()
            self._save_model()
            self._evaluate()
        except Exception as e:
            self._logger.error(f"An error occurred during the training pipeline: {e}")
            raise

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
            Configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
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

    def _setup_environment(self):
        """
        Configure the training environment.

        This typically involves setting seeds for reproducibility, configuring
        GPU or CPU usage, and other environment-level settings.

        Uses the `setup_environment(self._config)` method of the injected training
        environment class to apply the environment configuration.
        """
        self._training_env.setup_environment(self._config)

    def _setup_model(self):
        """
        Instantiate or configure the model and related components.

        This method should:
        - Parse model hyperparameters from `self._config`.
        - Create the model (e.g., Keras or PyTorch model).
        - (Optional) Compile the model with an optimizer, loss function, and metrics.

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to define model setup logic.
        """
        raise NotImplementedError

    def _build_model(self):
        """
        Build or initialize model parameters.

        Typical tasks might include:
        - Running a dummy forward pass to initialize weights.
        - Logging or printing the model summary.

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to define how model building
            and initialization are performed.
        """
        raise NotImplementedError

    def _setup_callbacks(self):
        """
        Define training callbacks or hooks.

        Typical tasks might include:
        - Setting up TensorBoard logging, early stopping, or custom callbacks.
        - Initializing checkpoint managers or automatic schedulers.

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to define callback setup logic.
        """
        raise NotImplementedError

    def _train(self):
        """
        Execute the training loop.

        Typical tasks might include:
        - Iterating over the training dataset for multiple epochs.
        - Monitoring training metrics and losses.
        - Optionally validating the model on a validation dataset each epoch.

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to define the core training logic.
        """
        raise NotImplementedError

    def _save_model(self):
        """
        Save the trained model to disk.

        Typical tasks might include:
        - Saving the full model in a framework-specific format (e.g.,
          TensorFlow SavedModel, PyTorch checkpoint).
        - Exporting to alternative formats (e.g., ONNX).
        - Saving additional artifacts like tokenizers, config files, etc.

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to define model-saving logic.
        """
        raise NotImplementedError

    def _evaluate(self):
        """
        Evaluate the trained model on a test dataset.

        Typical tasks might include:
        - Loading or preparing the test dataset.
        - Running inference to compute loss, accuracy, or other metrics.
        - Logging or storing evaluation results.

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to define how evaluation is performed.
        """
        raise NotImplementedError

    @property
    def config(self) -> Dict[str, Any]:
        """
        Access the loaded configuration dictionary.

        Returns
        -------
        Dict[str, Any]
            The configuration dictionary loaded from `config_path`.
        """
        return self._config

import logging
from abc import ABC, abstractmethod

class BaseSupervisedDataLoader(ABC):
    """
    A base class for data loading and splitting in supervised learning tasks.

    This class defines a generic interface for reading raw data from files,
    splitting the data into train/validation/test sets, and creating datasets
    suitable for training workflows.

    Concrete subclasses should implement the abstract methods to handle
    domain-specific details.
    """
    def __init__(
        self,
        test_split: float,
        validation_split: float,
        logger: logging.Logger,
    ):
        """
        Initialize the BaseDataLoader with paths to input (X) and target (Y) data,
        along with train/validation/test split ratios.

        Parameters
        ----------
        test_split : float
            Fraction of the total dataset to allocate for testing (0 < test_split < 1).
        validation_split : float
            Fraction of the total dataset to allocate for validation (0 < validation_split < 1).
        logger : logging.Logger
            A logger instance for logging messages and diagnostic information.
        """
        self._test_split = test_split
        self._validation_split = validation_split
        self._logger = logger

        if not (0 < self._test_split < 1 and 0 < self._validation_split < 1):
            raise ValueError("`test_split` and `validation_split` must be between 0 and 1.")

        total_split = self._test_split + self._validation_split
        if not (0 < total_split < 1):
            raise ValueError("The sum of `test_split` and `validation_split` must be between 0 and 1.")

        self._train_split = 1.0 - total_split

        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

    @abstractmethod
    def setup_datasets(self):
        """
        Abstract method for loading, processing, and splitting data.

        Subclasses must implement this method to perform all data preparation
        steps and populate the internal `_train_dataset`, `_valid_dataset`,
        and `_test_dataset` attributes.
        """
        raise NotImplementedError

    def get_train_dataset(self):
        """
        Retrieve the training dataset.

        This method should return a previously split or created dataset
        containing the training portion of the data.

        Returns
        -------
        Any
            The training dataset. The exact type depends on the library
            or framework in use.
        """
        if self._train_dataset is None:
            raise RuntimeError("Dataset not set up. Call `setup_datasets()` first.")

        return self._train_dataset

    def get_valid_dataset(self):
        """
        Retrieve the validation dataset.

        This method should return the validation portion of the data, useful
        for model performance monitoring during training.

        Returns
        -------
        Any
            The validation dataset. The exact type depends on the library
            or framework in use.
        """
        if self._valid_dataset is None:
            raise RuntimeError("Dataset not set up. Call `setup_datasets()` first.")

        return self._valid_dataset

    def get_test_dataset(self):
        """
        Retrieve the test dataset.

        This method should return the test portion of the data, to be used
        for final evaluation of the model.

        Returns
        -------
        Any
            The test dataset. The exact type depends on the library
            or framework in use.
        """
        if self._test_dataset is None:
            raise RuntimeError("Dataset not set up. Call `setup_datasets()` first.")

        return self._test_dataset

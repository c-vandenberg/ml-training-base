import logging
from typing import Any

class BaseSupervisedDataLoader:
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
        x_data_file_path: str,
        y_data_file_path: str,
        test_split: float,
        validation_split: float,
        logger: logging.Logger,
    ):
        """
        Initialize the BaseDataLoader with paths to input (X) and target (Y) data,
        along with train/validation/test split ratios.

        Parameters
        ----------
        x_data_file_path : str
            File path to the input (X) data file.
        y_data_file_path : str
            File path to the target/label (Y) data file.
        test_split : float
            Fraction of the total dataset to allocate for testing (0 < test_split < 1).
        validation_split : float
            Fraction of the total dataset to allocate for validation (0 < validation_split < 1).
        logger : logging.Logger
            A logger instance for logging messages and diagnostic information.
        """
        self._x_data_file_path = x_data_file_path
        self._y_data_file_path = y_data_file_path
        self._test_split = test_split
        self._validation_split = validation_split
        self._logger = logger

        if not (0 < self._test_split < 1 and 0 < self._validation_split < 1):
            raise ValueError("`test_split` and `validation_split` must be between 0 and 1.")

        total_split = self._test_split + self._validation_split
        if not (0 < total_split < 1):
            raise ValueError("The sum of `test_split` and `validation_split` must be between 0 and 1.")

        self._train_split = 1.0 - total_split

    def load_data(self):
        """
        Load raw data from the specified file paths.

        This method is responsible for reading the contents of
        `self._x_data_file_path` and `self._y_data_file_path`,
        performing any necessary parsing or preprocessing.

        Raises
        ------
        NotImplementedError
            This method must be overridden in a subclass to implement the
            actual data loading logic.
        """
        raise NotImplementedError

    def split_data(self):
        """
        Split the loaded data into train, validation, and test sets.

        This method should read in-memory data from the results of
        `load_data()`, apply the provided `test_split` and
        `validation_split`, and store the partitions internally.

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to define specific splitting
            logic (e.g., random shuffling, stratification, etc.).
        """
        raise NotImplementedError

    def get_dataset(self, data: Any, training=True):
        """
        Convert raw data into a format suitable for model consumption.

        Parameters
        ----------
        data : Any
            The data subset (e.g., a NumPy array, pandas DataFrame, or
            other structure) to be converted into a dataset.
        training : bool, optional
            Whether the dataset is intended for training. A subclass may
            choose to apply different transformations (e.g., data augmentation)
            if `training` is True.

        Returns
        -------
        Any
            A dataset in the framework-specific format (e.g., a `tf.data.Dataset`
            or a PyTorch `DataLoader`). Subclasses must specify the return type
            more concretely.

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to implement data format conversion
            logic.
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

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to return the training dataset.
        """
        raise NotImplementedError

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

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to return the validation dataset.
        """
        raise NotImplementedError

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

        Raises
        ------
        NotImplementedError
            Must be overridden in a subclass to return the test dataset.
        """
        raise NotImplementedError

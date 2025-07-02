import os
import logging


class LevelFilter(logging.Filter):
    """
    A custom log filter that allows only records of a specific level to pass.
    """
    def __init(self, level: int):
        super().__init__()
        self._level = level

    def filter(self, record):
        return record.levelno == self._level


def configure_multi_level_logger(
    name: str = __name__,
    log_dir: str = './logs',
    base_level=logging.DEBUG
) -> logging.Logger:
    """
    Configures a module-specific logger that writes to separate files for each log level.

    Parameters
    ----------
    name : str, optional
        The name for the logger instance.
    log_dir : str, optional
        The directory where log files will be saved.
    base_level : int, optional
        The lowest level of message the logger will process.

    Returns
    -------
    logging.Logger
        The fully configured logger instance.
    """
    # Ensure the log directory exists.
    os.makedirs(log_dir, exist_ok=True)

    # 1. Get the logger instance and set its base level
    logger = logging.getLogger(name)
    logger.setLevel(base_level)

    # Prevent messages from propagating to the root logger
    logger.propagate = False

    # Define the log files and the level for each
    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR
    }

    # Standard format for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 2. Create a filtered handler for each log level
    for level_name, level_int in log_levels.items():
        # 2.1. Create the file handler for this level
        file_path = os.path.join(log_dir, f'{level_name}.log')
        handler = logging.FileHandler(file_path, mode='w')
        handler.setLevel(level_int)
        handler.setFormatter(formatter)

        # 2.2. Create a filter that ONLY allows messages of this specific level
        level_filter = LevelFilter(level_int)
        handler.addFilter(level_filter)

        # 2.3. Add the configured handler to the main logger
        logger.addHandler(handler)

    # 3. Add a console handler to see all messages on the screen
    console_handler = logging.StreamHandler()
    console_handler.setLevel(base_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def configure_single_level_logger(log_path: str) -> logging.Logger:
    """
    Configures a module-specific logger that writes to a single file for each log level.

    Parameters
    ----------
    log_path : str, optional
        The directory where log files will be saved.

    Returns
    ----------
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        # Console handler for level INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # File handler for level DEBUG and above
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

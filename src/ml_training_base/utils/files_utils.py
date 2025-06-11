import os
import logging
from typing import List, Optional


def write_strings_to_file(
    file_path: str,
    str_list: List[str],
    logger: logging.Logger,
    content_name: str = "lines",
    log_interval: Optional[int] = 1000
):
    """
    Writes a list of strings to a file, with each string on a new line.

    This utility function creates the necessary directories, writes the content,
    and logs progress at specified intervals.

    Parameters
    ----------
    file_path : str
        The full path to the output file where the lines will be written.
    str_list : List[str]
        A list of strings to be written to the file.
    logger : logging.Logger
        A configured logger instance for status messages.
    content_name : str, optional
        A descriptive name for the content being written, used for logging.
        (default is "lines").
    log_interval : Optional[int], optional
        The interval at which to log progress (e.g., every 1000 lines).
        (default is 1000).

    """
    logger.info(f'Starting writing {content_name} to file...')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as file:
        total: int = len(str_list)
        for idx, line in enumerate(str_list):
            file.write(line + '\n')

            if (idx + 1) % log_interval == 0:
                logger.info(f'Written {idx + 1} / {total} {content_name} to file.')

    logger.info(f'Writing {total} {content_name} to file completed successfully.')

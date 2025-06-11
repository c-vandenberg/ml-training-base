import sqlite3
import logging
from typing import List, Optional, TypeVar, Generic

T = TypeVar('T')


class BaseDataPreprocessor(Generic[T]):
    """
    A generic base class providing data preprocessing utilities for machine
    learning pipelines.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger if logger else logging.getLogger(__name__)

    def concatenate_data(self, dataset_a: List[T], dataset_b: List[T]) -> List[T]:
        """
        Concatenates two lists of the same generic type.

        Parameters
        ----------
        dataset_a : List[T]
            The first dataset.
        dataset_b : List[T]
            The second dataset to be appended to the first.

        Returns
        -------
        List[T]
            The concatenated list.
        """
        if dataset_a is not None and dataset_b is not None:
            self._logger.info("Concatenating datasets.")
            self._logger.info(f"Dataset A size before concatenation: {len(dataset_a)}")
            dataset_a.extend(dataset_b)
            self._logger.info(f"Dataset size after concatenation: {len(dataset_a)}")

        return dataset_a

    def deduplicate_in_memory(self, data: List[T], content_name: str = "items") -> List[T]:
        """
        Deduplicates a list of items using an in-memory set.

        This method is fully generic and works with any hashable data type.

        Parameters
        ----------
        data : List[T]
            The list of items to deduplicate.
        content_name : str, optional
            A descriptive name for the items being processed, for logging.

        Returns
        -------
        List[T]
            A list containing only the unique items from the input data.
        """
        self._logger.info(f"Starting in-memory deduplication for {len(data)} {content_name}.")
        seen = set()
        unique_items = []
        for item in data:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        self._logger.info(f"Deduplication completed. Found {len(unique_items)} unique {content_name}.")

        return unique_items

    def deduplicate_on_disk(
        self,
        data: List[T],
        db_path: str = 'unique_items.db',
        content_name: str = "items",
        batch_size: int = 1000,
        log_interval: int = 1000
    ) -> List[T]:
        """
        Deduplicates a list using a SQLite database.

        Parameters
        ----------
        data : List[T]
            The list of items to deduplicate.
        db_path : str, optional
            Path to the SQLite database file.
        content_name : str, optional
            A descriptive name for the items being processed, for logging.
        batch_size : int, optional
            The number of items to insert into the database in each transaction.
        log_interval : int, optional
            The interval at which to log progress (e.g., every 100,000 items).

        Returns
        -------
        List[T]
            A list containing all unique items found in the database.

        Notes
        -----
        This method relies on converting items to strings for database storage.
        It is best suited for primitive types like `str`, `int`, or `float`.
        Using it with complex objects may result in loss of type information,
        as all retrieved items will be strings.
        """
        self._logger.info(f"Starting SQLite-based deduplication for {len(data)} {content_name}.")
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS unique_items (item TEXT PRIMARY KEY)")
            cursor = conn.cursor()

            total = len(data)
            current_idx = 0
            batch_number = 1

            while current_idx < total:
                batch_data = data[current_idx: current_idx + batch_size]

                # Convert items to strings for SQL insertion
                batch_for_sql = [(str(item),) for item in batch_data]

                try:
                    cursor.execute("BEGIN TRANSACTION;")
                    cursor.executemany("INSERT OR IGNORE INTO unique_items (item) VALUES (?)", batch_for_sql)
                    conn.commit()
                    self._logger.debug(f"Batch {batch_number} inserted with {len(batch_data)} {content_name}.")
                except sqlite3.Error as e:
                    self._logger.error(f"SQLite error on batch {batch_number}: {e}. Skipping batch.")
                    conn.rollback()

                current_idx += len(batch_data)
                batch_number += 1

                if current_idx % log_interval == 0:
                    self._logger.info(f"Processed {current_idx}/{total} {content_name}.")

            self._logger.info("Database insertion phase completed successfully.")
        finally:
            conn.close()

        return self._extract_from_db(db_path=db_path, content_name=content_name)

    def _extract_from_db(self, db_path: str, content_name: str = "items") -> List[T]:
        """
        Extracts all unique items from the SQLite database.
        """
        self._logger.info(f"Extracting unique {content_name} from the SQLite database.")
        unique_items: List[T] = []
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT item FROM unique_items")
            rows = cursor.fetchall()
            self._logger.debug(f"Fetched {len(rows)} unique reactions from the database.")

            # `cursor.fetchall()` returns list of tuples (e.g. [('datapoint-1', ), ('datapoint-2', )):
                # Access first element of each tuple during list comprehension to unpack tuples and give
                # list of datapoints.
            unique_items = [item[0] for item in rows]
            self._logger.info(f"Assigned {len(unique_items)} unique {content_name} to in-memory datasets.")
        except sqlite3.Error as e:
            self._logger.error(f"SQLite error during extraction: {e}")
        finally:
            conn.close()

        return unique_items

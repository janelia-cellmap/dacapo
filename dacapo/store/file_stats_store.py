from .stats_store import StatsStore
from .converter import converter
from dacapo.experiments import TrainingStats, TrainingIterationStats
from dacapo.experiments import ValidationScores, ValidationIterationScores
from typing import List

import logging
import pickle
from upath import UPath as Path

logger = logging.getLogger(__name__)


class FileStatsStore(StatsStore):
    """A File based store for run statistics. Used to store and retrieve training statistics and validation scores.

    The store is organized as follows:
    - A directory for training statistics, with a subdirectory for each run. Each run directory contains a pickled list of TrainingIterationStats objects.
    - A directory for validation scores, with a subdirectory for each run. Each run directory contains a pickled list of ValidationIterationScores objects.

    Attributes:
    - path: The root directory for the store.
    - training_stats: The directory for training statistics.
    - validation_scores: The directory for validation scores.

    Methods:
    - store_training_stats(run_name, stats): Store the training statistics for a run.
    - retrieve_training_stats(run_name): Retrieve the training statistics for a run.
    - store_validation_iteration_scores(run_name, scores): Store the validation scores for a run.
    - retrieve_validation_iteration_scores(run_name): Retrieve the validation scores for a run.
    - delete_training_stats(run_name): Delete the training statistics for a run.

    Note: The store does not support concurrent access. It is intended for use in single-threaded applications.


    """

    def __init__(self, path):
        """
        Initializes a new instance of the FileStatsStore class.

        Args:
            path (str): The path to the file.
        Raises:
            ValueError: If the path is invalid.
        Examples:
            >>> store = FileStatsStore("store")

        """
        print(f"Creating FileStatsStore:\n\tpath    : {path}")

        self.path = Path(path)

        self.__open_collections()
        self.__init_db()

    def store_training_stats(self, run_name, stats):
        """
        Stores the training statistics for a specific run.

        Args:
            run_name (str): The name of the run.
            stats (Stats): The training statistics to be stored.
        Raises:
            ValueError: If the run name is invalid.
        Examples:
            >>> store.store_training_stats("run1", stats)
        Notes:
            - If the training statistics for the given run already exist in the database, the method will compare the
              existing statistics with the new statistics and update or overwrite them accordingly.
            - If the new statistics go further than the existing statistics, the method will update the statistics from
              the last stored iteration.
            - If the new statistics are behind the existing statistics, the method will overwrite the existing statistics.
        """
        existing_stats = self.__read_training_stats(run_name)

        store_from_iteration = 0

        if existing_stats.trained_until() > 0:
            if stats.trained_until() > 0:
                # both current stats and DB contain data
                if stats.trained_until() > existing_stats.trained_until():
                    # current stats go further than the one in DB
                    store_from_iteration = existing_stats.trained_until()
                    logger.debug(
                        f"Updating training stats of run {run_name} after iteration {store_from_iteration}"
                    )
                else:
                    # current stats are behind DB--drop DB
                    existing_stats = None
                    logger.warning(
                        f"Overwriting previous training stats for run {run_name}"
                    )
                    self.__delete_training_stats(run_name)

        # store all new stats
        self.__store_training_stats(
            existing_stats, stats, store_from_iteration, stats.trained_until(), run_name
        )

    def retrieve_training_stats(self, run_name):
        """
        Retrieve the training statistics for a specific run.

        Parameters:
            run_name (str): The name of the run for which to retrieve the training statistics.

        Returns:
            dict: A dictionary containing the training statistics for the specified run.
        """
        return self.__read_training_stats(run_name)

    def store_validation_iteration_scores(self, run_name, scores):
        """
        Stores the validation scores for a specific run.

        Args:
            run_name (str): The name of the run.
            scores (Scores): The validation scores to be stored.
        Raises:
            ValueError: If the run name is invalid.
        Examples:
            >>> store.store_validation_iteration_scores("run1", scores)
        Notes:
            - If the validation scores for the given run already exist in the database, the method will compare the
              existing scores with the new scores and update or overwrite them accordingly.
            - If the new scores go further than the existing scores, the method will update the scores from
              the last stored iteration.
            - If the new scores are behind the existing scores, the method will overwrite the existing scores.
        """
        existing_scores = self.__read_validation_iteration_scores(run_name)
        store_from_iteration, drop_db = scores.compare(existing_scores)

        if drop_db:
            # current scores are behind DB--drop DB
            logger.warn(f"Overwriting previous validation scores for run {run_name}")
            self.__delete_validation_iteration_scores(run_name)

        if store_from_iteration > 0:
            print(
                f"Updating validation scores of run {run_name} after iteration {store_from_iteration}"
            )

        self.__store_validation_iteration_scores(
            scores, store_from_iteration, scores.validated_until() + 1, run_name
        )

    def retrieve_validation_iteration_scores(self, run_name):
        """
        Retrieve the validation iteration scores for a given run.

        Args:
            run_name (str): The name of the run for which to retrieve the validation iteration scores.
        Returns:
            list: A list of validation iteration scores.
        Raises:
            ValueError: If the run name is invalid.
        Examples:
            >>> store.retrieve_validation_iteration_scores("run1")

        """
        return self.__read_validation_iteration_scores(run_name)

    def delete_training_stats(self, run_name: str) -> None:
        """
        Deletes the training stats for a specific run.

        Args:
            run_name (str): The name of the run for which to delete the training stats.
        Raises:
            ValueError: If the run name is invalid.
        Examples:
            >>> store.delete_training_stats("run1")

        """
        self.__delete_training_stats(run_name)

    def __store_training_stats(self, existing_stats, stats, begin, end, run_name):
        """
        Store the training statistics for a specific run.

        Args:
            existing_stats (Stats): The statistics object containing the training stats that are already stored.
            stats (Stats): The statistics object containing the training stats.
            begin (int): The starting index of the iteration stats to store.
            end (int): The ending index of the iteration stats to store.
            run_name (str): The name of the run.
        Raises:
            ValueError: If the run name is invalid.
        Examples:
            >>> store.__store_training_stats(stats, 0, 100, "run1")

        """
        docs = converter.unstructure(stats.iteration_stats[begin:end])

        if docs:
            if existing_stats:
                # prepend existing stats to new stats
                docs = converter.unstructure(existing_stats.iteration_stats) + docs

            for doc in docs:
                doc.update({"run_name": run_name})

            file_store = self.training_stats / run_name
            with file_store.open("wb") as fd:
                pickle.dump(docs, fd)

    def __read_training_stats(self, run_name):
        """
        Read the training statistics for a given run.

        Args:
            run_name (str): The name of the run for which to read the training statistics.
        Returns:
            TrainingStats: The training statistics for the run.
        Raises:
            ValueError: If the run name is invalid.
        Examples:
            >>> store.__read_training_stats("run1")
        """
        file_store = self.training_stats / run_name
        if file_store.exists():
            with file_store.open("rb") as fd:
                docs = pickle.load(fd)
        else:
            docs = []
        stats = TrainingStats(converter.structure(docs, List[TrainingIterationStats]))
        return stats

    def __delete_training_stats(self, run_name):
        """
        Deletes the training stats file for a given run.

        Args:
            run_name (str): The name of the run for which to delete the training stats.
        Raises:
            ValueError: If the run name is invalid.
        Examples:
            >>> store.__delete_training_stats("run1")
        """
        file_store = self.training_stats / run_name
        if file_store.exists():
            file_store.unlink()

    def __store_validation_iteration_scores(
        self, validation_scores: ValidationScores, begin: int, end: int, run_name: str
    ) -> None:
        """
        Store the validation iteration scores.

        Args:
            validation_scores (ValidationScores): The validation scores object.
            begin (int): The starting iteration index.
            end (int): The ending iteration index.
            run_name (str): The name of the run.
        Raises:
            ValueError: If the run name is invalid.
        Examples:
            >>> store.__store_validation_iteration_scores(validation_scores, 0, 100, "run1")

        """
        docs = [
            converter.unstructure(scores)
            for scores in validation_scores.scores
            if scores.iteration < end
        ]
        for doc in docs:
            doc.update({"run_name": run_name})

        if docs:
            file_store = self.validation_scores / run_name
            with file_store.open("wb") as fd:
                pickle.dump(docs, fd)

    def __read_validation_iteration_scores(self, run_name):
        """
        Read the validation iteration scores for a given run.

        Args:
            run_name (str): The name of the run for which to read the validation iteration scores.
        Returns:
            ValidationScores: The validation iteration scores for the run.
        Raises:
            ValueError: If the run name is invalid.
        Examples:
            >>> store.__read_validation_iteration_scores("run1")
        """
        file_store = self.validation_scores / run_name
        if file_store.exists():
            with file_store.open("rb") as fd:
                docs = pickle.load(fd)
        else:
            docs = []
        scores = converter.structure(docs, List[ValidationIterationScores])
        return scores

    def __delete_validation_iteration_scores(self, run_name):
        """
        Delete the validation iteration scores for a given run.

        Args:
            run_name (str): The name of the run for which to delete the validation iteration scores.
        Raises:
            ValueError: If the run name is invalid.
        Examples:
            >>> store.__delete_validation_iteration_scores("run1")

        """
        file_store = self.validation_scores / run_name
        if file_store.exists():
            file_store.unlink()

    def __init_db(self):
        """
        Initialize the database for the file stats store.

        This method creates the necessary directories for storing training statistics and validation scores.

        Raises:
            ValueError: If the path is invalid.
        Examples:
            >>> store.__init_db()
        """

        pass

    def __open_collections(self):
        """
        Open the collections for the file stats store.

        This method initializes the directories for storing training statistics and validation scores.

        Raises:
            ValueError: If the path is invalid.
        Examples:
            >>> store.__open_collections()
        """
        self.training_stats = self.path / "training_stats"
        self.training_stats.mkdir(exist_ok=True, parents=True)
        self.validation_scores = self.path / "validation_scores"
        self.validation_scores.mkdir(exist_ok=True, parents=True)

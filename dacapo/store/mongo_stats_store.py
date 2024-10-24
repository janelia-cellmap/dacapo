from .stats_store import StatsStore
from pymongo import MongoClient, ASCENDING
from .converter import converter
from dacapo.experiments import TrainingStats, TrainingIterationStats
from dacapo.experiments import ValidationScores, ValidationIterationScores

import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class MongoStatsStore(StatsStore):
    """A MongoDB store for run statistics. Used to store and retrieve training
    statistics and validation scores.

    Attributes:
        db_host (str): The host of the MongoDB database.
        db_name (str): The name of the MongoDB database.
        client (MongoClient): The MongoDB client.
        database (Database): The MongoDB database.
        training_stats (Collection): The collection for training statistics.
        validation_scores (Collection): The collection for validation scores.
    Methods:
        store_training_stats(run_name, stats): Store the training stats of a given run.
        retrieve_training_stats(run_name): Retrieve the training stats for a given run.
        store_validation_iteration_scores(run_name, scores): Store the validation iteration scores of a given run.
        retrieve_validation_iteration_scores(run_name): Retrieve the validation iteration scores for a given run.
        delete_training_stats(run_name): Delete the training stats associated with a specific run.
    Notes:
        The MongoStatsStore uses the 'training_stats' and 'validation_scores' collections to store the statistics.

    """

    def __init__(self, db_host, db_name):
        """
        Initialize the MongoStatsStore with the given host and database name.

        Args:
            db_host (str): The host of the MongoDB database.
            db_name (str): The name of the MongoDB database.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
        Notes:
            The MongoStatsStore will connect to the MongoDB database at the given host.

        """
        print(
            f"Creating MongoStatsStore:\n\thost    : {db_host}\n\tdatabase: {db_name}"
        )

        self.db_host = db_host
        self.db_name = db_name

        self.client = MongoClient(self.db_host)
        self.database = self.client[self.db_name]
        self.__open_collections()
        self.__init_db()

    def store_training_stats(self, run_name: str, stats: TrainingStats):
        """
        Store the training statistics for a specific run.

        Args:
            run_name (str): The name of the run.
            stats (TrainingStats): The training statistics to be stored.
        Raises:
            ValueError: If the training statistics are already stored.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> stats = TrainingStats()
            >>> store.store_training_stats(run_name, stats)
        Notes:
            The training statistics are stored in the 'training_stats' collection.
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
                    logger.warn(
                        f"Overwriting previous training stats for run {run_name}"
                    )
                    self.__delete_training_stats(run_name)

        # store all new stats
        self.__store_training_stats(
            stats, store_from_iteration, stats.trained_until(), run_name
        )

    def retrieve_training_stats(
        self, run_name: str, subsample: bool = False
    ) -> TrainingStats:
        """
        Retrieve the training statistics for a given run.

        Args:
            run_name (str): The name of the run.
            subsample (bool, optional): Whether to subsample the training statistics. Defaults to False.
        Returns:
            TrainingStats: The training statistics for the specified run.
        Raises:
            ValueError: If the training statistics are not available.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> store.retrieve_training_stats(run_name)
        Notes:
            The training statistics are retrieved from the 'training_stats' collection.
        """
        return self.__read_training_stats(run_name, subsample=subsample)

    def store_validation_iteration_scores(
        self, run_name: str, scores: ValidationScores
    ):
        """
        Stores the validation iteration scores for a given run.

        Args:
            run_name (str): The name of the run.
            scores (ValidationScores): The validation scores to store.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> scores = ValidationScores()
            >>> store.store_validation_iteration_scores(run_name, scores)
        Notes:
            The validation iteration scores are stored in the 'validation_scores' collection.
        """

        existing_iteration_scores = self.__read_validation_iteration_scores(run_name)

        drop_db, store_from_iteration = scores.compare(existing_iteration_scores)

        if drop_db:
            # current scores are behind DB--drop DB
            logger.warn(f"Overwriting previous validation scores for run {run_name}")
            self.__delete_validation_scores(run_name)

        if store_from_iteration > 0:
            print(
                f"Updating validation scores of run {run_name} after iteration {store_from_iteration}"
            )

        self.__store_validation_iteration_scores(
            scores, store_from_iteration, scores.validated_until() + 1, run_name
        )

    def retrieve_validation_iteration_scores(
        self,
        run_name: str,
        subsample: bool = False,
        validation_interval: Optional[int] = None,
    ) -> List[ValidationIterationScores]:
        """
        Retrieve the validation iteration scores for a given run.

        Args:
            run_name (str): The name of the run.
            subsample (bool, optional): Whether to subsample the scores. Defaults to False.
            validation_interval (int, optional): The interval at which to retrieve the scores. Defaults to None.
        Returns:
            List[ValidationIterationScores]: A list of validation iteration scores.
        Raises:
            ValueError: If the validation iteration scores are not available.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> store.retrieve_validation_iteration_scores(run_name)
        Notes:
            The validation iteration scores are retrieved from the 'validation_scores' collection.
        """
        return self.__read_validation_iteration_scores(
            run_name, subsample=subsample, validation_interval=validation_interval
        )

    def __store_training_stats(
        self, stats: TrainingStats, begin: int, end: int, run_name: str
    ) -> None:
        """
        Store the training statistics in the database.

        Args:
            stats (TrainingStats): The training statistics to store.
            begin (int): The first iteration to store.
            end (int): The last iteration to store.
            run_name (str): The name of the run.
        Raises:
            ValueError: If the training statistics are already stored.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> stats = TrainingStats()
            >>> begin = 0
            >>> end = 1000
            >>> run_name = 'run_0'
            >>> store.__store_training_stats(stats, begin, end, run_name)
        Notes:
            The training statistics are stored in the 'training_stats' collection.

        """
        docs = converter.unstructure(stats.iteration_stats[begin:end])
        for doc in docs:
            doc.update({"run_name": run_name})

        if docs:
            self.training_stats.insert_many(docs)

    def __read_training_stats(
        self, run_name: str, subsample: bool = False
    ) -> TrainingStats:
        """
        Read training statistics from the MongoDB collection.

        Args:
            run_name (str): The name of the training run.
            subsample (bool, optional): Whether to subsample the statistics to get 1000 iterations. Defaults to False.
        Returns:
            TrainingStats: The training statistics.
        Raises:
            ValueError: If the training statistics are not available.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> store.__read_training_stats(run_name)
        Notes:
            The training statistics are read from the 'training_stats' collection.
        """
        filters: Dict[str, Any] = {"run_name": run_name}
        if subsample:
            # if possible subsample s.t. we get 1000 iterations
            iterations = list(
                self.training_stats.find(filters).sort("iteration", -1).limit(1)
            )
            if len(iterations) == 0:
                return TrainingStats()
            else:
                max_iteration = iterations[0]
                filters["iteration"] = {
                    "$mod": [(max_iteration["iteration"] + 999) // 1000, 0]
                }
        docs = list(self.training_stats.find(filters))
        if subsample and not docs[-1] == max_iteration:
            docs += [max_iteration]
        stats = TrainingStats(converter.structure(docs, List[TrainingIterationStats]))

        return stats

    def __delete_training_stats(self, run_name: str) -> None:
        """
        Delete training stats for a given run name.

        Args:
            run_name (str): The name of the run.
        Raises:
            ValueError: If the training statistics are not available.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> store.__delete_training_stats(run_name)
        Notes:
            The training statistics are deleted from the 'training_stats' collection.
        """
        self.training_stats.delete_many({"run_name": run_name})

    def __store_validation_iteration_scores(
        self,
        validation_scores: ValidationScores,
        begin: int,
        end: int,
        run_name: str,
    ) -> None:
        """
        Store the validation scores for a specific range of iterations.

        Args:
            validation_scores (ValidationScores): The validation scores object containing the scores to be stored.
            begin (int): The starting iteration (inclusive) for which the scores should be stored.
            end (int): The ending iteration (exclusive) for which the scores should be stored.
            run_name (str): The name of the run associated with the scores.
        Raises:
            ValueError: If the validation scores are already stored.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> validation_scores = ValidationScores()
            >>> begin = 0
            >>> end = 1000
            >>> run_name = 'run_0'
            >>> store.__store_validation_iteration_scores(validation_scores, begin, end, run_name)
        Notes:
            The validation scores are stored in the 'validation_scores' collection.
        """
        docs = [
            converter.unstructure(scores)
            for scores in validation_scores.scores
            if scores.iteration >= begin and scores.iteration < end
        ]
        for doc in docs:
            doc.update({"run_name": run_name})

        if docs:
            self.validation_scores.insert_many(docs)

    def __read_validation_iteration_scores(
        self,
        run_name: str,
        subsample: bool = False,
        validation_interval: Optional[int] = None,
    ) -> List[ValidationIterationScores]:
        """
        Read and retrieve validation iteration scores from the MongoDB collection.

        Args:
            run_name (str): The name of the run.
            subsample (bool, optional): Whether to subsample the scores. Defaults to False.
            validation_interval (int, optional): The interval at which to subsample the scores.
                Only applicable if subsample is True. Defaults to None.
        Returns:
            List[ValidationIterationScores]: A list of validation iteration scores.
        Raises:
            ValueError: If there is an error in processing the documents.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> store.__read_validation_iteration_scores(run_name)
        Notes:
            The validation iteration scores are read from the 'validation_scores' collection.
        """
        filters: Dict[str, Any] = {"run_name": run_name}
        if subsample:
            # if possible subsample s.t. we get 1000 iterations
            iterations = list(
                self.validation_scores.find(filters).sort("iteration", -1).limit(1)
            )
            if len(iterations) == 0:
                return []
            else:
                max_iteration = iterations[0]
                divisor = (max_iteration["iteration"] + 999) // 1000
                # round divisor down to nearest validation_interval
                divisor -= divisor % validation_interval
                # avoid using 0 as a divisor
                divisor = max(divisor, validation_interval)
                filters["iteration"] = {"$mod": [divisor, 0]}
        docs = list(self.validation_scores.find(filters))
        if subsample and not docs[-1] == max_iteration:
            docs += [max_iteration]
        try:
            scores = converter.structure(docs, List[ValidationIterationScores])
        except TypeError as e:
            # process each doc
            raise ValueError(docs[0]) from e
            scores = converter.structure(docs, List[ValidationIterationScores])
        return scores

    def delete_validation_scores(self, run_name: str) -> None:
        """
        Deletes the validation scores for a given run.

        Args:
            run_name (str): The name of the run for which validation scores should be deleted.
        Raises:
            ValueError: If the validation scores are not available.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> store.delete_validation_scores(run_name)
        Notes:
            The validation scores are deleted from the 'validation_scores' collection.
        """
        self.__delete_validation_scores(run_name)

    def __delete_validation_scores(self, run_name: str) -> None:
        """
        Delete validation scores for a given run name.

        Args:
            run_name (str): The name of the run.
        Raises:
            ValueError: If the validation scores are not available.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> store.__delete_validation_scores(run_name)
        Notes:
            The validation scores are deleted from the 'validation_scores' collection.

        """
        self.validation_scores.delete_many({"run_name": run_name})

    def delete_training_stats(self, run_name: str) -> None:
        """
        Deletes the training stats for a given run.

        Args:
            run_name (str): The name of the run for which training stats should be deleted.
        Raises:
            ValueError: If the training statistics are not available.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> store.delete_training_stats(run_name)
        Notes:
            The training statistics are deleted from the 'training_stats' collection.
        """
        self.__delete_training_stats(run_name)

    def __init_db(self):
        """
        Initialize the database by creating indexes for the training_stats and validation_scores collections.

        This method creates indexes on specific fields to improve query performance.

        Indexes created:
        - For training_stats collection:
            - run_name and iteration (unique index)
            - iteration
        - For validation_scores collection:
            - run_name, iteration, and dataset (unique index)
            - iteration
        Raises:
            ValueError: If the indexes cannot be created.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> store.__init_db()
        Notes:
            The indexes are created to improve query performance.
        """
        self.training_stats.create_index(
            [("run_name", ASCENDING), ("iteration", ASCENDING)],
            name="run_it",
            unique=True,
        )
        self.validation_scores.create_index(
            [("run_name", ASCENDING), ("iteration", ASCENDING), ("dataset", ASCENDING)],
            name="run_it_ds",
            unique=True,
        )
        self.training_stats.create_index([("iteration", ASCENDING)], name="it")
        self.validation_scores.create_index([("iteration", ASCENDING)], name="it")

    def __open_collections(self):
        """
        Opens the collections in the MongoDB database.

        This method initializes the `training_stats` and `validation_scores` collections
        in the MongoDB database.

        Raises:
            ValueError: If the collections are not available.
        Examples:
            >>> store = MongoStatsStore('localhost', 'dacapo')
            >>> store.__open_collections()
        Notes:
            The collections are used to store training statistics and validation scores.
        """
        self.training_stats = self.database["training_stats"]
        self.validation_scores = self.database["validation_scores"]

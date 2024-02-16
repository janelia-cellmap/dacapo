from .stats_store import StatsStore
from pymongo import MongoClient, ASCENDING
from .converter import converter
from dacapo.experiments import TrainingStats, TrainingIterationStats
from dacapo.experiments import ValidationScores, ValidationIterationScores

import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class MongoStatsStore(StatsStore):
    """
    The main class to interact with MongoDB for storing and retrieving
    training statistics and validation scores. This class directly interacts
    with the MongoDB client.

    Attributes:
        db_host: The host address of the MongoDB.
        db_name: The database name in MongoDB to where data will be stored.
        client: The MongoClient instance.
        database: The database instance of the specified database.
    """

    def __init__(self, db_host, db_name):
        """
        Create a new MongoDB store for keeping track of training statistics.

        Args:
            db_host: The host address of the MongoDB.
            db_name: The name of the database in MongoDB to where data will be stored.
        """
        logger.info(
            "Creating MongoStatsStore:\n\thost    : %s\n\tdatabase: %s",
            db_host,
            db_name,
        )

        self.db_host = db_host
        self.db_name = db_name

        self.client = MongoClient(self.db_host)
        self.database = self.client[self.db_name]
        self.__open_collections()
        self.__init_db()

    def store_training_stats(self, run_name: str, stats: TrainingStats):
        """
        Store the training statistics to the database.
        
        Args:
            run_name: A string denoting the name of the run.
            stats: An instance of TrainingStats containing the training statistics.
        """

    def retrieve_training_stats(
        self, run_name: str, subsample: bool = False
    ) -> TrainingStats:
        """
        Retrieve the training statistics from the database.
        
        Args:
            run_name: A string denoting the name of the run.
            subsample: A boolean indicating whether to subsample the data or not.

        Returns:
            An instance of TrainingStats containing the retrieved training statistics.
        """

    def store_validation_iteration_scores(
        self, run_name: str, scores: ValidationScores
    ):
        """
        Store the validation scores to the database.
        
        Args:
            run_name: A string denoting the name of the run.
            scores: An instance of ValidationScores containing the validation scores.
        """

    def retrieve_validation_iteration_scores(
        self,
        run_name: str,
        subsample: bool = False,
        validation_interval: Optional[int] = None,
    ) -> List[ValidationIterationScores]:
        """
        Retrieve the validation scores from the database.
        
        Args:
            run_name: A string denoting the name of the run.
            subsample: A boolean indicating whether to subsample the data or not.
            validation_interval: An integer specifying the validation interval.

        Returns:
            A list of ValidationIterationScores instances containing the retrieved validation scores.
        """

    def delete_validation_scores(self, run_name: str) -> None:
        """
        Delete the validation scores of a specific run from the database.
        
        Args:
            run_name: A string denoting the name of the run.
        """

    def delete_training_stats(self, run_name: str) -> None:
        """
        Delete the training statistics of a specific run from the database.
        
        Args:
            run_name: A string denoting the name of the run.
        """

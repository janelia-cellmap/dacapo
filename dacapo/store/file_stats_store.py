from .stats_store import StatsStore
from pymongo import MongoClient, ASCENDING
from .converter import converter
from dacapo.experiments import TrainingStats, TrainingIterationStats
from dacapo.experiments import ValidationScores, ValidationIterationScores
from typing import List

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class FileStatsStore(StatsStore):
    """A File based store for run statistics. Used to store and retrieve training
    statistics and validation scores.
    """

    def __init__(self, path):

        logger.info("Creating MongoStatsStore:\n\tpath    : %s", path)

        self.path = Path(path)

        self.__open_collections()
        self.__init_db()

    def store_training_stats(self, run_name, stats):

        existing_stats = self.__read_training_stats(run_name)

        store_from_iteration = 0

        if existing_stats.trained_until() > 0:

            if stats.trained_until() > 0:

                # both current stats and DB contain data
                if stats.trained_until() > existing_stats.trained_until():
                    # current stats go further than the one in DB
                    store_from_iteration = existing_stats.trained_until()
                    logger.info(
                        "Updating training stats of run %s after iteration %d",
                        run_name,
                        store_from_iteration,
                    )
                else:
                    # current stats are behind DB--drop DB
                    logger.warn(
                        "Overwriting previous training stats for run %s", run_name
                    )
                    self.__delete_training_stats(run_name)

        # store all new stats
        self.__store_training_stats(
            stats, store_from_iteration, stats.trained_until(), run_name
        )

    def retrieve_training_stats(self, run_name):

        return self.__read_training_stats(run_name)

    def store_validation_iteration_scores(self, run_name, scores):

        existing_iteration_scores = self.__read_validation_iteration_scores(run_name)
        store_from_iteration, drop_db = scores.compare(existing_iteration_scores)

        if drop_db:
            # current scores are behind DB--drop DB
            logger.warn("Overwriting previous validation scores for run %s", run_name)
            self.__delete_validation_iteration_scores(run_name)

        if store_from_iteration > 0:
            logger.info(
                "Updating validation scores of run %s after iteration " "%d",
                run_name,
                store_from_iteration,
            )

        self.__store_validation_iteration_scores(
            scores, store_from_iteration, scores.validated_until() + 1, run_name
        )

    def retrieve_validation_iteration_scores(self, run_name):

        return self.__read_validation_iteration_scores(run_name)

    def __store_training_stats(self, stats, begin, end, run_name):

        docs = converter.unstructure(stats.iteration_stats[begin:end])
        for doc in docs:
            doc.update({"run_name": run_name})

        if docs:
            file_store = self.training_stats / run_name
            pickle.dump(docs, file_store.open("wb"))

    def __read_training_stats(self, run_name):

        file_store = self.training_stats / run_name
        if file_store.exists():
            docs = pickle.load(file_store.open("rb"))
        else:
            docs = []
        stats = TrainingStats(converter.structure(docs, List[TrainingIterationStats]))
        return stats

    def __delete_training_stats(self, run_name):

        file_store = self.training_stats / run_name
        if file_store.exists():
            file_store.unlink()

    def __store_validation_iteration_scores(
        self, validation_scores, begin, end, run_name
    ):

        docs = [
            converter.unstructure(scores)
            for scores in validation_scores.iteration_scores
            if scores.iteration < end
        ]
        for doc in docs:
            doc.update({"run_name": run_name})

        if docs:
            file_store = self.validation_scores / run_name
            pickle.dump(docs, file_store.open("wb"))

    def __read_validation_iteration_scores(self, run_name):

        file_store = self.validation_scores / run_name
        if file_store.exists():
            docs = pickle.load(file_store.open("rb"))
        else:
            docs = []
        scores = converter.structure(docs, List[ValidationIterationScores])
        return scores

    def __delete_validation_iteration_scores(self, run_name):

        file_store = self.validation_scores / run_name
        if file_store.exists():
            file_store.unlink()

    def __init_db(self):
        pass

    def __open_collections(self):

        self.training_stats = self.path / "training_stats"
        self.training_stats.mkdir(exist_ok=True, parents=True)
        self.validation_scores = self.path / "validation_scores"
        self.validation_scores.mkdir(exist_ok=True, parents=True)

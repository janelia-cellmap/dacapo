from .stats_store import StatsStore
from pymongo import MongoClient, ASCENDING
from .converter import converter
from dacapo.experiments import TrainingStats, TrainingIterationStats
from dacapo.experiments import ValidationScores, ValidationIterationScores
from dacapo.experiments.tasks.evaluators import *
from typing import List
import logging
import time

logger = logging.getLogger(__name__)


class MongoStatsStore(StatsStore):
    """A MongoDB store for run statistics. Used to store and retrieve training
    statistics and validation scores.
    """

    def __init__(self, db_host, db_name):

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

    def retrieve_training_stats(self, run_name, subsample=False):

        return self.__read_training_stats(run_name, subsample=subsample)

    def store_validation_iteration_scores(self, run_name, scores):

        existing_iteration_scores = self.__read_validation_iteration_scores(run_name)

        store_from_iteration, drop_db = scores.compare(existing_iteration_scores)

        if drop_db:
            # current scores are behind DB--drop DB
            logger.warn("Overwriting previous validation scores for run %s", run_name)
            self.__delete_validation_scores(run_name)

        if store_from_iteration > 0:
            logger.info(
                "Updating validation scores of run %s after iteration " "%d",
                run_name,
                store_from_iteration,
            )

        self.__store_validation_iteration_scores(
            scores, store_from_iteration, scores.validated_until() + 1, run_name
        )

    def retrieve_validation_iteration_scores(
        self, run_name, subsample=False, validation_interval=None
    ):
        return self.__read_validation_iteration_scores(
            run_name, subsample=subsample, validation_interval=validation_interval
        )

    def __store_training_stats(self, stats, begin, end, run_name):

        docs = converter.unstructure(stats.iteration_stats[begin:end])
        for doc in docs:
            doc.update({"run_name": run_name})

        if docs:
            self.training_stats.insert_many(docs)

    def __read_training_stats(self, run_name, subsample=False):
        # TODO: using the converter to structure the training/validation stats is extremely slow.
        # (3e-5 seconds to get training stats, 6 seconds to convert)
        filters = {"run_name": run_name}
        if subsample:
            # if possible subsample s.t. we get 1000 iterations
            max_iteration = list(
                self.training_stats.find(filters).sort("iteration", -1).limit(1)
            )
            if len(max_iteration) == 0:
                return TrainingStats()
            else:
                max_iteration = max_iteration[0]
            filters["iteration"] = {
                "$mod": [(max_iteration["iteration"] + 999) // 1000, 0]
            }
        docs = list(self.training_stats.find(filters))
        if subsample and not docs[-1] == max_iteration:
            docs += [max_iteration]
        stats = TrainingStats(converter.structure(docs, List[TrainingIterationStats]))

        return stats

    def __delete_training_stats(self, run_name):

        self.training_stats.delete_many({"run_name": run_name})

    def __store_validation_iteration_scores(
        self, validation_scores, begin, end, run_name
    ):

        docs = [
            converter.unstructure(scores)
            for scores in validation_scores.iteration_scores
            if scores.iteration >= begin and scores.iteration < end
        ]
        for doc in docs:
            doc.update({"run_name": run_name})

        if docs:
            self.validation_scores.insert_many(docs)

    def __read_validation_iteration_scores(
        self, run_name, subsample=False, validation_interval=None
    ):
        # TODO: using the converter to structure the training/validation stats is extremely slow.
        # (5e-5 seconds to get validation stats, 3 seconds to convert)
        filters = {"run_name": run_name}
        if subsample:
            # if possible subsample s.t. we get 1000 iterations
            max_iteration = list(
                self.validation_scores.find(filters).sort("iteration", -1).limit(1)
            )
            if len(max_iteration) == 0:
                return ValidationScores()
            else:
                max_iteration = max_iteration[0]
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

    def delete_validation_scores(self, run_name):
        self.__delete_validation_scores(run_name)

    def __delete_validation_scores(self, run_name):

        self.validation_scores.delete_many({"run_name": run_name})

    def delete_training_stats(self, run_name):
        self.__delete_training_stats(run_name)

    def __init_db(self):

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

        self.training_stats = self.database["training_stats"]
        self.validation_scores = self.database["validation_scores"]

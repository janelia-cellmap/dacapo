from .stats_store import StatsStore
from pymongo import MongoClient, ASCENDING
from .converter import converter
from dacapo.experiments import TrainingStats, TrainingIterationStats
from dacapo.experiments import ValidationScores, ValidationIterationScores
from typing import List
import logging

logger = logging.getLogger(__name__)


class MongoStatsStore(StatsStore):
    """A MongoDB store for run statistics. Used to store and retrieve training
    statistics and validation scores.
    """

    def __init__(self, db_host, db_name):

        logger.info(
            "Creating MongoStatsStore:\n\thost    : %s\n\tdatabase: %s",
            db_host, db_name)

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
                        store_from_iteration)
                else:
                    # current stats are behind DB--drop DB
                    logger.warn(
                        "Overwriting previous training stats for run %s",
                        run_name)
                    self.__delete_training_stats(run_name)

        # store all new stats
        self.__store_training_stats(
            stats,
            store_from_iteration,
            stats.trained_until(),
            run_name)

    def retrieve_training_stats(self, run_name):

        return self.__read_training_stats(run_name)

    def store_validation_scores(self, run_name, scores):

        existing_scores = self.__read_validation_scores(run_name)
        existing_scores_until = existing_scores.validated_until()

        store_from_iteration = 0

        if existing_scores_until > 0:

            if scores.validated_until() > 0:

                # both current scores and DB contain data
                if scores.validated_until() > existing_scores_until:
                    # current scores go further than the one in DB
                    store_from_iteration = existing_scores_until
                    logger.info(
                        "Updating validation scores of run %s after iteration "
                        "%d",
                        run_name,
                        store_from_iteration)
                else:
                    # current scores are behind DB--drop DB
                    logger.warn(
                        "Overwriting previous validation scores for run %s",
                        run_name)
                    self.__delete_validation_scores(run_name)

        self.__store_validation_scores(
            scores,
            store_from_iteration,
            scores.validated_until(),
            run_name)

    def retrieve_validation_scores(self, run_name):

        return self.__read_validation_scores(run_name)

    def __store_training_stats(self, stats, begin, end, run_name):

        docs = converter.unstructure(stats.iteration_stats[begin:end])
        for doc in docs:
            doc.update({'run_name': run_name})

        if docs:
            self.training_stats.insert_many(docs)

    def __read_training_stats(self, run_name):

        docs = self.training_stats.find({'run_name': run_name})
        stats = TrainingStats(
            converter.structure(
                docs,
                List[TrainingIterationStats]))
        return stats

    def __delete_training_stats(self, run_name):

        self.training_stats.delete_many({'run_name': run_name})

    def __store_validation_scores(
            self,
            validation_scores,
            begin,
            end,
            run_name):

        docs = [
            converter.unstructure(scores)
            for scores in validation_scores.iteration_scores
            if scores.iteration >= begin or scores.iteration < end
        ]
        for doc in docs:
            doc.update({'run_name': run_name})

        if docs:
            self.validation_scores.insert_many(docs)

    def __read_validation_scores(self, run_name):

        docs = self.validation_scores.find({'run_name': run_name})
        scores = ValidationScores(
            converter.structure(
                docs,
                List[ValidationIterationScores]))
        return scores

    def __delete_validation_scores(self, run_name):

        self.validation_scores.delete_many({'run_name': run_name})

    def __init_db(self):

        self.training_stats.create_index(
            [('run_name', ASCENDING), ('iteration', ASCENDING)],
            name='run_it',
            unique=True
        )
        self.validation_scores.create_index(
            [('run_name', ASCENDING), ('iteration', ASCENDING)],
            name='run_it',
            unique=True
        )

    def __open_collections(self):

        self.training_stats = self.database['training_stats']
        self.validation_scores = self.database['validation_scores']

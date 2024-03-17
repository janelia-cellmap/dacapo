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
    """

    def __init__(self, db_host, db_name):
        print(
            "Creating MongoStatsStore:\n\thost    : %s\n\tdatabase: %s" % db_host,
            db_name,
        )

        self.db_host = db_host
        self.db_name = db_name

        self.client = MongoClient(self.db_host)
        self.database = self.client[self.db_name]
        self.__open_collections()
        self.__init_db()

    def store_training_stats(self, run_name: str, stats: TrainingStats):
        existing_stats = self.__read_training_stats(run_name)

        store_from_iteration = 0

        if existing_stats.trained_until() > 0:
            if stats.trained_until() > 0:
                # both current stats and DB contain data
                if stats.trained_until() > existing_stats.trained_until():
                    # current stats go further than the one in DB
                    store_from_iteration = existing_stats.trained_until()
                    print(
                        "Updating training stats of run %s after iteration %d"
                        % run_name,
                        store_from_iteration,
                    )
                else:
                    # current stats are behind DB--drop DB
                    logger.warn(
                        "Overwriting previous training stats for run %s" % run_name
                    )
                    self.__delete_training_stats(run_name)

        # store all new stats
        self.__store_training_stats(
            stats, store_from_iteration, stats.trained_until(), run_name
        )

    def retrieve_training_stats(
        self, run_name: str, subsample: bool = False
    ) -> TrainingStats:
        return self.__read_training_stats(run_name, subsample=subsample)

    def store_validation_iteration_scores(
        self, run_name: str, scores: ValidationScores
    ):
        existing_iteration_scores = self.__read_validation_iteration_scores(run_name)

        drop_db, store_from_iteration = scores.compare(existing_iteration_scores)

        if drop_db:
            # current scores are behind DB--drop DB
            logger.warn("Overwriting previous validation scores for run %s" % run_name)
            self.__delete_validation_scores(run_name)

        if store_from_iteration > 0:
            print(
                "Updating validation scores of run %s after iteration " "%d" % run_name,
                store_from_iteration,
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
        return self.__read_validation_iteration_scores(
            run_name, subsample=subsample, validation_interval=validation_interval
        )

    def __store_training_stats(
        self, stats: TrainingStats, begin: int, end: int, run_name: str
    ) -> None:
        docs = converter.unstructure(stats.iteration_stats[begin:end])
        for doc in docs:
            doc.update({"run_name": run_name})

        if docs:
            self.training_stats.insert_many(docs)

    def __read_training_stats(
        self, run_name: str, subsample: bool = False
    ) -> TrainingStats:
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
        self.training_stats.delete_many({"run_name": run_name})

    def __store_validation_iteration_scores(
        self,
        validation_scores: ValidationScores,
        begin: int,
        end: int,
        run_name: str,
    ) -> None:
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
        self.__delete_validation_scores(run_name)

    def __delete_validation_scores(self, run_name: str) -> None:
        self.validation_scores.delete_many({"run_name": run_name})

    def delete_training_stats(self, run_name: str) -> None:
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

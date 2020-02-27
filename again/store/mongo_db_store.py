from again.training_stats import TrainingStats
from again.validation_scores import ValidationScores
from pymongo import MongoClient, ASCENDING
import configargparse
import json

parser = configargparse.ArgParser(
    default_config_files=['~/.config/again', './again.conf'])
parser.add(
    '--mongo_db_host',
    help="Name of the MongoDB host for stats and scores")
parser.add(
    '--mongo_db_name',
    help="Name of the MongoDB database for stats and scores")


class MongoDbStore:

    def __init__(self):
        """Create a MongoDB sync. Used to sync runs, tasks, models,
        optimizers, trainig stats, and validation scores."""

        options = parser.parse_known_args()[0]
        self.db_host = options.mongo_db_host
        self.db_name = options.mongo_db_name

        self.client = MongoClient(self.db_host)
        self.database = self.client[self.db_name]
        self.__open_collections()
        self.__init_db()

    def sync_run(
            self,
            run,
            exclude_training_stats=False,
            exclude_validation_scores=False):

        self.__sync_run(run)
        self.__sync_task_config(run.task_config)
        self.__sync_model_config(run.model_config)
        self.__sync_optimizer_config(run.optimizer_config)

    def store_training_stats(self, run):

        stats = run.training_stats
        existing_stats = self.__read_training_stats(run.id)

        store_from_iteration = 0

        if existing_stats.trained_until() > 0:

            if stats.trained_until() > 0:

                # both current stats and DB contain data
                if stats.trained_until() > existing_stats.trained_until():
                    # current stats go further than the one in DB
                    store_from_iteration = existing_stats.trained_until()
                else:
                    # current stats are behind DB--drop DB
                    self.__delete_training_stats(run.id)

        # store all new stats
        self.__store_training_stats(
            stats,
            store_from_iteration,
            stats.trained_until(),
            run.id)

    def read_training_stats(self, run):

        run.training_stats = self.__read_training_stats(run.id)

    def store_validation_scores(self, run):

        scores = run.validation_scores
        existing_scores = self.__read_validation_scores(run.id)

        store_from_iteration = 0

        if existing_scores.validated_until() > 0:

            if scores.validated_until() > 0:

                # both current scores and DB contain data
                if scores.validated_until() > \
                        existing_scores.validated_until():
                    # current scores go further than the one in DB
                    store_from_iteration = existing_scores.validated_until()
                else:
                    # current scores are behind DB--drop DB
                    self.__delete_validation_scores(run.id)

        self.__store_validation_scores(
                scores,
                store_from_iteration,
                scores.validated_until(),
                run.id)

    def read_validation_scores(self, run):

        run.validation_scores = self.__read_validation_scores(run.id)

    def __sync_run(self, run):

        run_doc = run.to_dict()
        existing = list(
            self.runs.find(
                {'id': run.id}, {'_id': False}))

        if existing:

            stored_run = existing[0]

            if not self.__same_doc(
                    run_doc,
                    stored_run,
                    ignore=['started', 'stopped', 'num_parameters']):
                raise RuntimeError(
                    f"Data for run {run.id} does not match already synced "
                    f"entry. Found\n\n{stored_run}\n\nin DB, but was "
                    f"given\n\n{run_doc}")

            # stored and existing are the same, except maybe for started and
            # stopped timestamp

            update_db = False
            if stored_run['started'] is None and run.started is not None:
                update_db = True
            if stored_run['stopped'] is None and run.stopped is not None:
                update_db = True

            update_current = False
            if stored_run['started'] is not None and run.started is None:
                update_current = True
            if stored_run['stopped'] is not None and run.stopped is None:
                update_current = True

            if update_db and update_current:
                raise RuntimeError(
                    f"Start and stop time of run {run.id} do not match "
                    f"already synced entry. Found\n\n{stored_run}\n\nin "
                    f"DB, but was given\n\n{run_doc}")

            if update_db:
                self.runs.update({'id': run.id}, run_doc)
            elif update_current:
                run.started = stored_run['started']
                run.stopped = stored_run['stopped']

        else:

            self.runs.insert(run_doc)

    def __sync_task_config(self, task_config):
        self.__save_insert(self.tasks, task_config.to_dict())

    def __sync_model_config(self, model_config):
        self.__save_insert(
            self.models,
            model_config.to_dict())

    def __sync_optimizer_config(self, optimizer_config):
        self.__save_insert(
            self.optimizers,
            optimizer_config.to_dict())

    def __store_training_stats(self, stats, begin, end, run_id):

        docs = []
        for i in range(begin, end):
            docs.append({
                'run': run_id,
                'iteration': int(stats.iterations[i]),
                'loss': float(stats.losses[i]),
                'time': float(stats.times[i])
            })
        if docs:
            self.training_stats.insert_many(docs)

    def __read_training_stats(self, run_id):

        stats = TrainingStats()
        docs = self.training_stats.find({'run': run_id})
        for doc in docs:
            stats.add_training_iteration(
                doc['iteration'],
                doc['loss'],
                doc['time'])
        return stats

    def __delete_training_stats(self, run_id):
        self.training_stats.delete_many({'run': run_id})

    def __store_validation_scores(self, validation_scores, begin, end, run_id):

        docs = []
        for idx, iteration in enumerate(validation_scores.iterations):
            if iteration < begin or iteration >= end:
                continue
            docs.append({
                'run': run_id,
                'iteration': int(iteration),
                'scores': {
                    sample: {
                        score: values[idx]
                        for score, values in scores.items()
                    }
                    for sample, scores in
                    validation_scores.sample_scores.items()
                }
            })

        if docs:
            self.validation_scores.insert_many(docs)

    def __read_validation_scores(self, run_id):

        validation_scores = ValidationScores()
        docs = self.validation_scores.find({'run': run_id})
        for doc in docs:
            validation_scores.add_validation_iteration(
                doc['iteration'],
                {
                    sample: {
                        score: value
                        for score, value in scores.items()
                    }
                    for sample, scores in doc['scores'].items()
                })

        return validation_scores

    def __delete_validation_scores(self, run_id):
        self.validation_scores.delete_many({'run': run_id})

    def __save_insert(self, collection, data, ignore=None):

        existing = collection.find_one_and_update(
                filter={'id': data['id']},
                update={'$set': data},
                upsert=True,
                projection={'_id': False})

        if existing:

            if not self.__same_doc(existing, data, ignore):
                raise RuntimeError(
                    f"Data for {data['id']} does not match already synced "
                    f"entry. Found\n\n{existing}\n\nin DB, but was "
                    f"given\n\n{data}")

        return existing

    def __same_doc(self, a, b, ignore=None):

        if ignore:
            a = dict(a)
            b = dict(b)
            for key in ignore:
                if key in a:
                    del a[key]
                if key in b:
                    del b[key]

        # JSONify both and compare
        a = json.loads(json.dumps(a))
        b = json.loads(json.dumps(b))

        return a == b

    def __init_db(self):

        self.runs.create_index(
            [
                ('id', ASCENDING),
                ('repetition', ASCENDING)
            ],
            name='id_rep',
            unique=True)
        self.tasks.create_index(
            [
                ('id', ASCENDING)
            ],
            name='id',
            unique=True)
        self.models.create_index(
            [
                ('id', ASCENDING)
            ],
            name='id',
            unique=True)
        self.optimizers.create_index(
            [
                ('id', ASCENDING)
            ],
            name='id',
            unique=True)
        self.training_stats.create_index(
            [
                ('run', ASCENDING),
                ('iteration', ASCENDING)
            ],
            name='run_it',
            unique=True)
        self.validation_scores.create_index(
            [
                ('run', ASCENDING),
                ('iteration', ASCENDING)
            ],
            name='run_it',
            unique=True)

    def __open_collections(self):
        '''Opens the node, edge, and meta collections'''

        self.runs = self.database['runs']
        self.tasks = self.database['tasks']
        self.models = self.database['models']
        self.optimizers = self.database['optimizers']
        self.training_stats = self.database['training_stats']
        self.validation_scores = self.database['validation_scores']

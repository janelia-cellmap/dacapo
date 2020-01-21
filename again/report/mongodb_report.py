import json
from pymongo import MongoClient, ASCENDING


class MongoDbReport:

    def __init__(self, db_host, db_name, run_config):
        """Create a MongoDB report for this run. Creates documents for tasks,
        models, optimizers, and this run if they don't exist."""

        self.db_host = db_host
        self.db_name = db_name
        self.run_config = run_config
        self.run_repetition = None
        self._iterations = []

        self.__connect()
        self.__open_db()
        self.__open_collections()
        self.__init_db()

        self.__store_task()
        self.__store_model()
        self.__store_optimizer()
        self.__store_run()

    def add_model_size(self, num_params):

        self.models.update_one(
            {'id': self.run_config.model.id},
            {'$set': {'num_params': num_params}})

    def add_training_iteration(self, iteration, loss, time):

        self._iterations.append({
            'run': self.run_config.id,
            'repetition': self.run_repetition,
            'iteration': iteration,
            'loss': float(loss),
            'time': time})

        if len(self._iterations) > 100:
            self.train.insert_many(self._iterations)
            self._iterations = []

    def __store_task(self):

        self.__save_insert(self.tasks, self.run_config.task.to_dict())

    def __store_model(self):

        self.__save_insert(
            self.models,
            self.run_config.model.to_dict(),
            ignore='num_params')

    def __store_optimizer(self):

        self.__save_insert(
            self.optimizers,
            self.run_config.optimizer.to_dict())

    def __store_run(self):

        existing = list(
            self.runs.find(
                {'id': self.run_config.id}, {'_id': False}))
        max_repetition = 0
        if existing:
            for doc in existing:
                max_repetition = max(max_repetition, doc['repetition'])

        self.run_repetition = max_repetition + 1
        run_doc = self.run_config.to_dict()
        run_doc['repetition'] = self.run_repetition

        self.runs.insert(run_doc)

    def __save_insert(self, collection, data, ignore=None):

        existing = list(
            collection.find(
                {'id': data['id']}, {'_id': False}))

        if len(existing) > 0:

            if ignore:
                if ignore in existing[0]:
                    del existing[0][ignore]
                if ignore in data:
                    data = dict(data)
                    del data[ignore]

            if not self.__same_doc(existing[0], data):
                raise RuntimeError(
                    f"Data for {data['id']} does not match already stored "
                    f"entry. Found\n\n{existing[0]}\n\nin DB, but was "
                    f"given\n\n{data}")
        else:
            collection.insert(data)

    def __same_doc(self, a, b):

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
        self.train.create_index(
            [
                ('run', ASCENDING),
                ('iteration', ASCENDING),
                ('repetition', ASCENDING)
            ],
            name='run_it_rep',
            unique=True)

    def __connect(self):
        """Init the MongDb client."""

        self.client = MongoClient(self.db_host)

    def __open_db(self):
        '''Opens Mongo database'''

        self.database = self.client[self.db_name]

    def __open_collections(self):
        '''Opens the node, edge, and meta collections'''

        self.runs = self.database['runs']
        self.tasks = self.database['tasks']
        self.models = self.database['models']
        self.optimizers = self.database['optimizers']
        self.train = self.database['train']

    def __disconnect(self):
        """Closes the mongo client and removes references to all collections
        and databases"""

        self.database = None
        self.client.close()
        self.client = None

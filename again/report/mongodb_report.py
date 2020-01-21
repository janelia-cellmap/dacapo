import json
from pymongo import MongoClient, ASCENDING


class MongoDbReport:

    def __init__(self, db_host, db_name, run_config):
        """Create a MongoDB report for this run. Creates documents for tasks,
        models, optimizers, and this run if they don't exist."""

        self.db_host = db_host
        self.db_name = db_name
        self.run_config = run_config
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
            {'name': self.run_config.model.name},
            {'$set': {'num_params': num_params}})

    def add_training_iteration(self, iteration, loss, time):

        self._iterations.append({
            'run': self.run_config.name,
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

        self.__save_insert(self.optimizers, self.run_config.optimizer.to_dict())

    def __store_run(self):

        self.__save_insert(self.runs, self.run_config.to_dict())

    def __save_insert(self, collection, data, ignore=None):

        existing = list(
            collection.find(
                {'name': data['name']}, {'_id': False}))

        if len(existing) > 0:

            if ignore:
                if ignore in existing[0]:
                    del existing[0][ignore]
                if ignore in data:
                    data = dict(data)
                    del data[ignore]

            if not self.__same_doc(existing[0], data):
                raise RuntimeError(
                    f"Data for {data['name']} does not match already stored "
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
                ('id', ASCENDING)
            ],
            name='id',
            unique=True)
        self.runs.create_index(
            [
                ('name', ASCENDING)
            ],
            name='name',
            unique=True)
        self.tasks.create_index(
            [
                ('name', ASCENDING)
            ],
            name='name',
            unique=True)
        self.models.create_index(
            [
                ('name', ASCENDING)
            ],
            name='name',
            unique=True)
        self.optimizers.create_index(
            [
                ('name', ASCENDING)
            ],
            name='name',
            unique=True)
        self.train.create_index(
            [
                ('iteration', ASCENDING)
            ],
            name='name',
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

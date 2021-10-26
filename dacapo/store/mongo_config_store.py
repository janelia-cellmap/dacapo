from .config_store import ConfigStore, DuplicateNameError
from .converter import converter
from dacapo.experiments import RunConfig
from dacapo.experiments.architectures import ArchitectureConfig
from dacapo.experiments.datasplits import DataSplitConfig
from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
from dacapo.experiments.tasks import TaskConfig
from dacapo.experiments.trainers import TrainerConfig
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
import logging

logger = logging.getLogger(__name__)


class MongoConfigStore(ConfigStore):
    """A MongoDB store for configurations. Used to store and retrieve
    configurations for runs, tasks, architectures, trainers, and datasets.
    """

    def __init__(self, db_host, db_name):

        logger.info(
            "Creating MongoConfigStore:\n\thost    : %s\n\tdatabase: %s",
            db_host, db_name)

        self.db_host = db_host
        self.db_name = db_name

        self.client = MongoClient(self.db_host)
        self.database = self.client[self.db_name]
        self.__open_collections()
        self.__init_db()

    def store_run_config(self, run_config):

        run_doc = converter.unstructure(run_config)
        self.__save_insert(self.runs, run_doc)

    def retrieve_run_config(self, run_name):

        run_doc = self.runs.find_one(
            {"name": run_name},
            projection={"_id": False})
        return converter.structure(run_doc, RunConfig)

    def retrieve_run_config_names(self):

        runs = self.runs.find(
            {},
            projection={"_id": False, "name": True})
        return list([run["name"] for run in runs])

    def store_task_config(self, task_config):

        task_doc = converter.unstructure(task_config)
        self.__save_insert(self.tasks, task_doc)

    def retrieve_task_config(self, task_name):

        task_doc = self.tasks.find_one(
            {"name": task_name},
            projection={"_id": False})
        return converter.structure(task_doc, TaskConfig)

    def retrieve_task_config_names(self):

        tasks = self.tasks.find(
            {},
            projection={"_id": False, "name": True})
        return list([task["name"] for task in tasks])

    def store_architecture_config(self, architecture_config):

        architecture_doc = converter.unstructure(architecture_config)
        self.__save_insert(self.architectures, architecture_doc)

    def retrieve_architecture_config(self, architecture_name):

        architecture_doc = self.architectures.find_one(
            {"name": architecture_name},
            projection={"_id": False})
        return converter.structure(architecture_doc, ArchitectureConfig)

    def retrieve_architecture_config_names(self):

        architectures = self.architectures.find(
            {},
            projection={"_id": False, "name": True})
        return list([architecture["name"] for architecture in architectures])

    def store_trainer_config(self, trainer_config):

        trainer_doc = converter.unstructure(trainer_config)
        self.__save_insert(self.trainers, trainer_doc)

    def retrieve_trainer_config(self, trainer_name):

        trainer_doc = self.trainers.find_one(
            {"name": trainer_name},
            projection={"_id": False})
        return converter.structure(trainer_doc, TrainerConfig)

    def retrieve_trainer_config_names(self):

        trainers = self.trainers.find(
            {},
            projection={"_id": False, "name": True})
        return list([trainer["name"] for trainer in trainers])

    def store_datasplit_config(self, datasplit_config):

        datasplit_doc = converter.unstructure(datasplit_config)
        self.__save_insert(self.datasplits, datasplit_doc)

    def retrieve_datasplit_config(self, datasplit_name):

        datasplit_doc = self.datasplits.find_one(
            {"name": datasplit_name},
            projection={"_id": False})
        return converter.structure(datasplit_doc, DataSplitConfig)

    def retrieve_datasplit_config_names(self):

        datasplits = self.datasplits.find(
            {},
            projection={"_id": False, "name": True})
        return list([datasplit["name"] for datasplit in datasplits])

    def store_array_config(self, array_config):

        array_doc = converter.unstructure(array_config)
        self.__save_insert(self.arrays, array_doc)

    def retrieve_array_config(self, array_name):

        array_doc = self.arrays.find_one(
            {"name": array_name},
            projection={"_id": False})
        return converter.structure(array_doc, ArrayConfig)

    def retrieve_array_config_names(self):

        arrays = self.arrays.find(
            {},
            projection={"_id": False, "name": True})
        return list([array["name"] for array in arrays])

    def __save_insert(self, collection, data, ignore=None):

        name = data['name']

        try:

            collection.insert_one(dict(data))

        except DuplicateKeyError:

            existing = collection.find(
                {'name': name},
                projection={'_id': False})[0]

            if not self.__same_doc(existing, data, ignore):

                raise DuplicateNameError(
                    f"Data for {name} does not match already stored "
                    f"entry. Found\n\n{existing}\n\nin DB, but was "
                    f"given\n\n{data}"
                )

    def __same_doc(self, a, b, ignore=None):

        if ignore:
            a = dict(a)
            b = dict(b)
            for key in ignore:
                if key in a:
                    del a[key]
                if key in b:
                    del b[key]

        return a == b

    def __init_db(self):

        self.users.create_index(
            [("username", ASCENDING)],
            name="username",
            unique=True)

        self.runs.create_index(
            [("name", ASCENDING), ("repetition", ASCENDING)],
            name="name_rep",
            unique=True)

        self.tasks.create_index(
            [("name", ASCENDING)],
            name="name",
            unique=True)

        self.datasets.create_index(
            [("name", ASCENDING)],
            name="name",
            unique=True)

        self.architectures.create_index(
            [("name", ASCENDING)],
            name="name",
            unique=True)

        self.trainers.create_index(
            [("name", ASCENDING)],
            name="name",
            unique=True)

    def __open_collections(self):

        self.users = self.database["users"]
        self.runs = self.database["runs"]
        self.tasks = self.database["tasks"]
        self.datasets = self.database["datasets"]
        self.architectures = self.database["architectures"]
        self.trainers = self.database["trainers"]

from .config_store import ConfigStore, DuplicateNameError
from .converter import converter
from dacapo.experiments import RunConfig
from dacapo.experiments.architectures import ArchitectureConfig
from dacapo.experiments.datasplits import DataSplitConfig
from dacapo.experiments.datasplits.datasets import DatasetConfig
from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
from dacapo.experiments.tasks import TaskConfig
from dacapo.experiments.trainers import TrainerConfig
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

import logging
import bson

logger = logging.getLogger(__name__)


class MongoConfigStore(ConfigStore):
    """
    A MongoDB store for configurations. Used to store and retrieve
    configurations for runs, tasks, architectures, trainers, and datasets.

    Attributes:
        db_host (str): The host of the MongoDB database.
        db_name (str): The name of the MongoDB database.
        client (MongoClient): The MongoDB client.
        database (Database): The MongoDB database.
        users (Collection): The users collection.
        runs (Collection): The runs collection.
        tasks (Collection): The tasks collection.
        datasplits (Collection): The datasplits collection.
        datasets (Collection): The datasets collection.
        arrays (Collection): The arrays collection.
        architectures (Collection): The architectures collection.
        trainers (Collection): The trainers collection.
    Methods:
        store_run_config(run_config, ignore): Store the run configuration.
        retrieve_run_config(run_name): Retrieve the run configuration.
        delete_run_config(run_name): Delete the run configuration.
        retrieve_run_config_names(task_names, datasplit_names, architecture_names, trainer_names): Retrieve the names of the run configurations.
        store_task_config(task_config, ignore): Store the task configuration.
        retrieve_task_config(task_name): Retrieve the task configuration.
        retrieve_task_config_names(): Retrieve the names of the task configurations.
        store_architecture_config(architecture_config, ignore): Store the architecture configuration.
        retrieve_architecture_config(architecture_name): Retrieve the architecture configuration.
        retrieve_architecture_config_names(): Retrieve the names of the architecture configurations.
        store_trainer_config(trainer_config, ignore): Store the trainer configuration.
        retrieve_trainer_config(trainer_name): Retrieve the trainer configuration.
        retrieve_trainer_config_names(): Retrieve the names of the trainer configurations.
        store_datasplit_config(datasplit_config, ignore): Store the datasplit configuration.
        retrieve_datasplit_config(datasplit_name): Retrieve the datasplit configuration.
        retrieve_datasplit_config_names(): Retrieve the names of the datasplit configurations.
        store_dataset_config(dataset_config, ignore): Store the dataset configuration.
        retrieve_dataset_config(dataset_name): Retrieve the dataset configuration.
        retrieve_dataset_config_names(): Retrieve the names of the dataset configurations.
        store_array_config(array_config, ignore): Store the array configuration.
        retrieve_array_config(array_name): Retrieve the array configuration.
        retrieve_array_config_names(): Retrieve the names of the array configurations.
        __save_insert(collection, data, ignore): Save or insert a document into a collection.
        __same_doc(a, b, ignore): Check if two documents are the same.
        __init_db(): Initialize the database.
        __open_collections(): Open the collections.
    Notes:
        The store is initialized with the host and database name.
    """

    def __init__(self, db_host, db_name):
        """
        Initialize a MongoConfigStore object.

        Args:
            db_host (str): The host address of the MongoDB server.
            db_name (str): The name of the database to connect to.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
        """
        print(
            f"Creating MongoConfigStore:\n\thost    : {db_host}\n\tdatabase: {db_name}"
        )

        self.db_host = db_host
        self.db_name = db_name

        self.client = MongoClient(self.db_host)
        self.database = self.client[self.db_name]
        self.__open_collections()
        self.__init_db()

    def delete_config(self, database, config_name: str) -> None:
        """
        Deletes a configuration from the database.

        Args:
            database: The database object.
            config_name: The name of the configuration to delete.
        Raises:
            ValueError: If the configuration is not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> config_name = 'config_0'
            >>> store.delete_config(store.tasks, config_name)
        """
        database.delete_one({"name": config_name})

    def store_run_config(self, run_config, ignore=None):
        """
        Stores the run configuration in the MongoDB runs collection.

        Args:
            run_config (dict): The run configuration to be stored.
            ignore (list, optional): A list of fields to ignore during the storage process.
        Raises:
            DuplicateNameError: If the run configuration is already stored.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> run_config = {'name': 'run_0'}
            >>> store.store_run_config(run_config)
        """
        run_doc = converter.unstructure(run_config)
        self.__save_insert(self.runs, run_doc, ignore)

    def retrieve_run_config(self, run_name):
        """
        Retrieve the run configuration for a given run name.

        Args:
            run_name (str): The name of the run.
        Returns:
            RunConfig: The run configuration for the given run name.
        Raises:
            ValueError: If the run configuration is not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> store.retrieve_run_config(run_name)
        """
        run_doc = self.runs.find_one({"name": run_name}, projection={"_id": False})
        try:
            return converter.structure(run_doc, RunConfig)
        except TypeError as e:
            raise TypeError(f"Could not structure run: {run_name} as RunConfig!") from e

    def delete_run_config(self, run_name):
        """
        Delete a run configuration from the MongoDB collection.

        Args:
            run_name (str): The name of the run configuration to delete.
        Raises:
            ValueError: If the run configuration is not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> run_name = 'run_0'
            >>> store.delete_run_config(run_name)
        """
        self.runs.delete_one({"name": run_name})

    def retrieve_run_config_names(
        self,
        task_names=None,
        datasplit_names=None,
        architecture_names=None,
        trainer_names=None,
    ):
        """
        Retrieve the names of run configurations based on specified filters.

        Args:
            task_names (list, optional): List of task names to filter the run configurations. Defaults to None.
            datasplit_names (list, optional): List of datasplit names to filter the run configurations. Defaults to None.
            architecture_names (list, optional): List of architecture names to filter the run configurations. Defaults to None.
            trainer_names (list, optional): List of trainer names to filter the run configurations. Defaults to None.
        Returns:
            list: A list of run configuration names that match the specified filters.
        Raises:
            ValueError: If the run configurations are not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> task_names = ['task_0']
            >>> datasplit_names = ['datasplit_0']
            >>> architecture_names = ['architecture_0']
            >>> trainer_names = ['trainer_0']
            >>> store.retrieve_run_config_names(task_names, datasplit_names, architecture_names, trainer_names)
        """
        filters = {}
        if task_names is not None:
            filters["task_config.name"] = {"$in": task_names}
        if datasplit_names is not None:
            filters["datasplit_config.name"] = {"$in": datasplit_names}
        if architecture_names is not None:
            filters["architecture_config.name"] = {"$in": architecture_names}
        if trainer_names is not None:
            filters["trainer_config.name"] = {"$in": trainer_names}
        runs = self.runs.find(filters, projection={"_id": False, "name": True})
        return list([run["name"] for run in runs])

    def store_task_config(self, task_config, ignore=None):
        """
        Store the task configuration in the MongoDB tasks collection.

        Args:
            task_config (TaskConfig): The task configuration to be stored.
            ignore (list, optional): A list of fields to ignore during the storage process.
        Raises:
            DuplicateNameError: If the task configuration is already stored.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> task_config = TaskConfig(name='task_0')
            >>> store.store_task_config(task_config)
        """
        task_doc = converter.unstructure(task_config)
        self.__save_insert(self.tasks, task_doc, ignore)

    def retrieve_task_config(self, task_name):
        """
        Retrieve the task configuration for a given task name.

        Args:
            task_name (str): The name of the task.
        Returns:
            TaskConfig: The task configuration object.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> task_name = 'task_0'
            >>> store.retrieve_task_config(task_name)

        """
        task_doc = self.tasks.find_one({"name": task_name}, projection={"_id": False})
        return converter.structure(task_doc, TaskConfig)

    def retrieve_task_config_names(self):
        """
        Retrieve the names of all task configurations.

        Returns:
            A list of task configuration names.
        Raises:
            ValueError: If the task configurations are not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> store.retrieve_task_config_names()

        """
        tasks = self.tasks.find({}, projection={"_id": False, "name": True})
        return list([task["name"] for task in tasks])

    def store_architecture_config(self, architecture_config, ignore=None):
        """
        Store the architecture configuration in the MongoDB.

        Args:
            architecture_config (ArchitectureConfig): The architecture configuration to be stored.
            ignore (list, optional): List of fields to ignore during storage. Defaults to None.
        Raises:
            DuplicateNameError: If the architecture configuration is already stored.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> architecture_config = ArchitectureConfig(name='architecture_0')
            >>> store.store_architecture_config(architecture_config)
        """
        architecture_doc = converter.unstructure(architecture_config)
        self.__save_insert(self.architectures, architecture_doc, ignore)

    def retrieve_architecture_config(self, architecture_name):
        """
        Retrieve the architecture configuration for the given architecture name.

        Args:
            architecture_name (str): The name of the architecture.
        Returns:
            ArchitectureConfig: The architecture configuration object.
        Raises:
            ValueError: If the architecture configuration is not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> architecture_name = 'architecture_0'
            >>> store.retrieve_architecture_config(architecture_name)

        """
        architecture_doc = self.architectures.find_one(
            {"name": architecture_name}, projection={"_id": False}
        )
        return converter.structure(architecture_doc, ArchitectureConfig)

    def retrieve_architecture_config_names(self):
        """
        Retrieve the names of all architecture configurations.

        Returns:
            A list of architecture configuration names.
        Raises:
            ValueError: If the architecture configurations are not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> store.retrieve_architecture_config_names()

        """
        architectures = self.architectures.find(
            {}, projection={"_id": False, "name": True}
        )
        return list([architecture["name"] for architecture in architectures])

    def store_trainer_config(self, trainer_config, ignore=None):
        """
        Store the trainer configuration in the MongoDB.

        Args:
            trainer_config (TrainerConfig): The trainer configuration to be stored.
            ignore (list, optional): List of fields to ignore during storage. Defaults to None.
        Returns:
            DuplicateNameError: If the trainer configuration is already stored.
        Raises:
            DuplicateNameError: If the trainer configuration is already stored.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> trainer_config = TrainerConfig(name='trainer_0')
            >>> store.store_trainer_config(trainer_config)
        """
        trainer_doc = converter.unstructure(trainer_config)
        self.__save_insert(self.trainers, trainer_doc, ignore)

    def retrieve_trainer_config(self, trainer_name):
        """
        Retrieve the trainer configuration for the given trainer name.

        Args:
            trainer_name (str): The name of the trainer.
        Returns:
            TrainerConfig: The trainer configuration object.
        Raises:
            ValueError: If the trainer configuration is not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> trainer_name = 'trainer_0'
            >>> store.retrieve_trainer_config(trainer_name)
        """
        trainer_doc = self.trainers.find_one(
            {"name": trainer_name}, projection={"_id": False}
        )
        return converter.structure(trainer_doc, TrainerConfig)

    def retrieve_trainer_config_names(self):
        """
        Retrieve the names of all trainer configurations.

        Args:
            trainer_name (str): The name of the trainer.
        Returns:
            TrainerConfig: The trainer configuration object.
        Raises:
            ValueError: If the trainer configuration is not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> trainer_name = 'trainer_0'
            >>> store.retrieve_trainer_config(trainer_name)
        """
        trainers = self.trainers.find({}, projection={"_id": False, "name": True})
        return list([trainer["name"] for trainer in trainers])

    def store_datasplit_config(self, datasplit_config, ignore=None):
        """
        Store the datasplit configuration in the MongoDB.

        Args:
            datasplit_config (DataSplitConfig): The datasplit configuration to be stored.
            ignore (list, optional): List of fields to ignore during storage. Defaults to None.
        Returns:
            DuplicateNameError: If the datasplit configuration is already stored.
        Raises:
            DuplicateNameError: If the datasplit configuration is already stored.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> datasplit_config = DataSplitConfig(name='datasplit_0')
            >>> store.store_datasplit_config(datasplit_config)
        """
        datasplit_doc = converter.unstructure(datasplit_config)
        self.__save_insert(self.datasplits, datasplit_doc, ignore)

    def retrieve_datasplit_config(self, datasplit_name):
        """
        Retrieve the datasplit configuration for the given datasplit name.

        Args:
            datasplit_name (str): The name of the datasplit.
        Returns:
            DataSplitConfig: The datasplit configuration object.
        Raises:
            ValueError: If the datasplit configuration is not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> datasplit_name = 'datasplit_0'
            >>> store.retrieve_datasplit_config(datasplit_name)
        """
        datasplit_doc = self.datasplits.find_one(
            {"name": datasplit_name}, projection={"_id": False}
        )
        return converter.structure(datasplit_doc, DataSplitConfig)

    def retrieve_datasplit_config_names(self):
        """
        Retrieve the names of all datasplit configurations.

        Returns:
            A list of datasplit configuration names.
        Raises:
            ValueError: If the datasplit configurations are not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> store.retrieve_datasplit_config_names()
        """
        datasplits = self.datasplits.find({}, projection={"_id": False, "name": True})
        return list([datasplit["name"] for datasplit in datasplits])

    def store_dataset_config(self, dataset_config, ignore=None):
        """
        Store the dataset configuration in the MongoDB.

        Args:
            dataset_config (DatasetConfig): The dataset configuration to be stored.
            ignore (list, optional): List of fields to ignore during storage. Defaults to None.
        Returns:
            DuplicateNameError: If the dataset configuration is already stored.
        Raises:
            DuplicateNameError: If the dataset configuration is already stored.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> dataset_config = DatasetConfig(name='dataset_0')
            >>> store.store_dataset_config(dataset_config)

        """
        dataset_doc = converter.unstructure(dataset_config)
        self.__save_insert(self.datasets, dataset_doc, ignore)

    def retrieve_dataset_config(self, dataset_name):
        """
        Retrieve the dataset configuration for the given dataset name.

        Args:
            dataset_name (str): The name of the dataset.
        Returns:
            DatasetConfig: The dataset configuration object.
        Raises:
            ValueError: If the dataset configuration is not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> dataset_name = 'dataset_0'
            >>> store.retrieve_dataset_config(dataset_name)
        """
        dataset_doc = self.datasets.find_one(
            {"name": dataset_name}, projection={"_id": False}
        )
        return converter.structure(dataset_doc, DatasetConfig)

    def retrieve_dataset_config_names(self):
        """
        Retrieve the names of all dataset configurations.

        Returns:
            A list of dataset configuration names.
        Raises:
            ValueError: If the dataset configurations are not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> store.retrieve_dataset_config_names()

        """
        datasets = self.datasets.find({}, projection={"_id": False, "name": True})
        return list([dataset["name"] for dataset in datasets])

    def store_array_config(self, array_config, ignore=None):
        """
        Store the array configuration in the MongoDB.

        Args:
            array_config (ArrayConfig): The array configuration to be stored.
            ignore (list, optional): List of fields to ignore during storage. Defaults to None.
        Returns:
            DuplicateNameError: If the array configuration is already stored.
        Raises:
            DuplicateNameError: If the array configuration is already stored.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> array_config = ArrayConfig(name='array_0')
            >>> store.store_array_config(array_config)
        """
        array_doc = converter.unstructure(array_config)
        self.__save_insert(self.arrays, array_doc, ignore)

    def retrieve_array_config(self, array_name):
        """
        Retrieve the array configuration for the given array name.

        Args:
            array_name (str): The name of the array.
        Returns:
            ArrayConfig: The array configuration object.
        Raises:
            ValueError: If the array configuration is not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> array_name = 'array_0'
            >>> store.retrieve_array_config(array_name)

        """
        array_doc = self.arrays.find_one(
            {"name": array_name}, projection={"_id": False}
        )
        return converter.structure(array_doc, ArrayConfig)

    def retrieve_array_config_names(self):
        """
        Retrieve the names of all array configurations.

        Returns:
            A list of array configuration names.
        Raises:
            ValueError: If the array configurations are not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> store.retrieve_array_config_names()
        """
        arrays = self.arrays.find({}, projection={"_id": False, "name": True})
        return list([array["name"] for array in arrays])

    def __save_insert(self, collection, data, ignore=None):
        """
        Save and insert data into the specified collection.

        Args:
            collection (pymongo.collection.Collection): The collection to insert the data into.
            data (dict): The data to be inserted.
            ignore (list, optional): A list of keys to ignore when comparing existing and new data.
        Raises:
            DuplicateNameError: If the data for the given name does not match the already stored entry.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> collection = store.runs
            >>> data = {'name': 'run_0'}
            >>> store.__save_insert(collection, data)
        """
        name = data["name"]

        try:
            collection.insert_one(dict(data))

        except DuplicateKeyError:
            existing = collection.find({"name": name}, projection={"_id": False})[0]

            if not self.__same_doc(existing, data, ignore):
                raise DuplicateNameError(
                    f"Data for {name} does not match already stored "
                    f"entry. Found\n\n{existing}\n\nin DB, but was "
                    f"given\n\n{data}"
                )

    def __same_doc(self, a, b, ignore=None):
        """
        Check if two documents are the same.

        Args:
            a (dict): The first document.
            b (dict): The second document.
            ignore (list, optional): A list of fields to ignore during the comparison.
        Returns:
            bool: True if the documents are the same, False otherwise.
        Raises:
            ValueError: If the documents are not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> a = {'name': 'run_0'}
            >>> b = {'name': 'run_0'}
            >>> store.__same_doc(a, b)

        """
        if ignore:
            a = dict(a)
            b = dict(b)
            for key in ignore:
                if key in a:
                    del a[key]
                if key in b:
                    del b[key]

        bson_a = bson.encode(a)
        bson_b = bson.encode(b)

        return bson_a == bson_b

    def __init_db(self):
        """
        Initialize the MongoDB database.

        Raises:
            ValueError: If the collections are not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> store.__init_db()
        """
        self.users.create_index([("username", ASCENDING)], name="username", unique=True)

        self.runs.create_index(
            [("name", ASCENDING), ("repetition", ASCENDING)],
            name="name_rep",
            unique=True,
        )

        self.tasks.create_index([("name", ASCENDING)], name="name", unique=True)

        self.datasplits.create_index([("name", ASCENDING)], name="name", unique=True)

        self.datasets.create_index([("name", ASCENDING)], name="name", unique=True)

        self.arrays.create_index([("name", ASCENDING)], name="name", unique=True)

        self.architectures.create_index([("name", ASCENDING)], name="name", unique=True)

        self.trainers.create_index([("name", ASCENDING)], name="name", unique=True)

    def __open_collections(self):
        """
        Opens the collections in the MongoDB database.

        This method initializes the collection attributes for various entities such as users, runs, tasks, datasplits, datasets,
        arrays, architectures, and trainers. These attributes can be used to interact with the corresponding collections in the database.

        Raises:
            ValueError: If the collections are not available.
        Examples:
            >>> store = MongoConfigStore('localhost', 'dacapo')
            >>> store.__open_collections()
        """
        self.users = self.database["users"]
        self.runs = self.database["runs"]
        self.tasks = self.database["tasks"]
        self.datasplits = self.database["datasplits"]
        self.datasets = self.database["datasets"]
        self.arrays = self.database["arrays"]
        self.architectures = self.database["architectures"]
        self.trainers = self.database["trainers"]

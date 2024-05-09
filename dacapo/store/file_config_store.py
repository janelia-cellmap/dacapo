from .config_store import ConfigStore, DuplicateNameError
from .converter import converter
from dacapo.experiments import RunConfig
from dacapo.experiments.architectures import ArchitectureConfig
from dacapo.experiments.datasplits import DataSplitConfig
from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
from dacapo.experiments.tasks import TaskConfig
from dacapo.experiments.trainers import TrainerConfig

import logging
import yaml
from upath import UPath as Path

logger = logging.getLogger(__name__)


class FileConfigStore(ConfigStore):
    """
    A Local File based store for configurations. Used to store and retrieve
    configurations for runs, tasks, architectures, trainers, and datasplits.

    Attributes:
        path (Path): The path to the file.
    Methods:
        store_run_config(run_config, ignore=None): Stores the run configuration in the file config store.
        retrieve_run_config(run_name): Retrieve the run configuration for a given run name.
        retrieve_run_config_names(): Retrieve the names of the run configurations.
        store_task_config(task_config, ignore=None): Stores the task configuration in the file config store.
        retrieve_task_config(task_name): Retrieve the task configuration for a given task name.
        retrieve_task_config_names(): Retrieve the names of the task configurations.
        store_architecture_config(architecture_config, ignore=None): Stores the architecture configuration in the file config store.
        retrieve_architecture_config(architecture_name): Retrieve the architecture configuration for a given architecture name.
        retrieve_architecture_config_names(): Retrieve the names of the architecture configurations.
        store_trainer_config(trainer_config, ignore=None): Stores the trainer configuration in the file config store.
        retrieve_trainer_config(trainer_name): Retrieve the trainer configuration for a given trainer name.
        retrieve_trainer_config_names(): Retrieve the names of the trainer configurations.
        store_datasplit_config(datasplit_config, ignore=None): Stores the datasplit configuration in the file config store.
        retrieve_datasplit_config(datasplit_name): Retrieve the datasplit configuration for a given datasplit name.
        retrieve_datasplit_config_names(): Retrieve the names of the datasplit configurations.
        store_array_config(array_config, ignore=None): Stores the array configuration in the file config store.
        retrieve_array_config(array_name): Retrieve the array configuration for a given array name.
        retrieve_array_config_names(): Retrieve the names of the array configurations.
        __save_insert(collection, data, ignore=None): Saves the data to the collection.
        __load(collection, name): Loads the data
    Notes:
        The FileConfigStore is used to store and retrieve configurations for runs, tasks, architectures, trainers, and datasplits.
        The FileConfigStore is a local file based store for configurations.
    """

    def __init__(self, path):
        """
        Initializes a new instance of the FileConfigStore class.

        Args:
            path (str): The path to the file.
        Raises:
            ValueError: If the path is not a valid directory.
        Examples:
            >>> store = FileConfigStore("path/to/configs")
        """
        print(f"Creating FileConfigStore:\n\tpath: {path}")

        self.path = Path(path)

        self.__open_collections()
        self.__init_db()

    def store_run_config(self, run_config, ignore=None):
        """
        Stores the run configuration in the file config store.

        Args:
            run_config (RunConfig): The run configuration to store.
            ignore (list, optional): A list of keys to ignore when comparing the stored configuration with the new configuration. Defaults to None.
        Raises:
            DuplicateNameError: If a configuration with the same name already exists.
        Examples:
            >>> store.store_run_config(run_config)
        """
        run_doc = converter.unstructure(run_config)
        self.__save_insert(self.runs, run_doc, ignore)

    def retrieve_run_config(self, run_name):
        """
        Retrieve the run configuration for a given run name.

        Args:
            run_name (str): The name of the run configuration to retrieve.
        Returns:
            RunConfig: The run configuration object.
        Raises:
            KeyError: If the run name does not exist in the store.
        Examples:
            >>> run_config = store.retrieve_run_config("run1")

        """
        run_doc = self.__load(self.runs, run_name)
        return converter.structure(run_doc, RunConfig)

    def retrieve_run_config_names(self):
        """
        Retrieve the names of the run configurations.

        Returns:
            A list of run configuration names.
        Raises:
            KeyError: If no run configurations are stored.
        Examples:
            >>> run_names = store.retrieve_run_config_names()

        """
        return [f.name[:-5] for f in self.runs.iterdir()]

    def store_task_config(self, task_config, ignore=None):
        """
        Stores the task configuration in the file config store.

        Args:
            task_config (TaskConfig): The task configuration to store.
            ignore (list, optional): A list of keys to ignore when comparing the stored configuration with the new configuration. Defaults to None.
        Raises:
            DuplicateNameError: If a configuration with the same name already exists.
        Examples:
            >>> store.store_task_config(task_config)

        """
        task_doc = converter.unstructure(task_config)
        self.__save_insert(self.tasks, task_doc, ignore)

    def retrieve_task_config(self, task_name):
        """
        Retrieve the task configuration for a given task name.

        Args:
            task_name (str): The name of the task configuration to retrieve.
        Returns:
            TaskConfig: The task configuration object.
        Raises:
            KeyError: If the task name does not exist in the store.
        Examples:
            >>> task_config = store.retrieve_task_config("task1")

        """
        task_doc = self.__load(self.tasks, task_name)
        return converter.structure(task_doc, TaskConfig)

    def retrieve_task_config_names(self):
        """
        Retrieve the names of the task configurations.

        Returns:
            A list of task configuration names.
        Raises:
            KeyError: If no task configurations are stored.
        Examples:
            >>> task_names = store.retrieve_task_config_names()
        """
        return [f.name[:-5] for f in self.tasks.iterdir()]

    def store_architecture_config(self, architecture_config, ignore=None):
        """
        Stores the architecture configuration in the file config store.

        Args:
            architecture_config (ArchitectureConfig): The architecture configuration to store.
            ignore (list, optional): A list of keys to ignore when comparing the stored configuration with the new configuration. Defaults to None.
        Raises:
            DuplicateNameError: If a configuration with the same name already exists.
        Examples:
            >>> store.store_architecture_config(architecture_config)
        """
        architecture_doc = converter.unstructure(architecture_config)
        self.__save_insert(self.architectures, architecture_doc, ignore)

    def retrieve_architecture_config(self, architecture_name):
        """
        Retrieve the architecture configuration for a given architecture name.

        Args:
            architecture_name (str): The name of the architecture configuration to retrieve.
        Returns:
            ArchitectureConfig: The architecture configuration object.
        Raises:
            KeyError: If the architecture name does not exist in the store.
        Examples:
            >>> architecture_config = store.retrieve_architecture_config("architecture1")
        """
        architecture_doc = self.__load(self.architectures, architecture_name)
        return converter.structure(architecture_doc, ArchitectureConfig)

    def retrieve_architecture_config_names(self):
        """
        Retrieve the names of the architecture configurations.

        Returns:
            A list of architecture configuration names.
        Raises:
            KeyError: If no architecture configurations are stored.
        Examples:
            >>> architecture_names = store.retrieve_architecture_config_names()
        """
        return [f.name[:-5] for f in self.architectures.iterdir()]

    def store_trainer_config(self, trainer_config, ignore=None):
        """
        Stores the trainer configuration in the file config store.

        Args:
            trainer_config (TrainerConfig): The trainer configuration to store.
            ignore (list, optional): A list of keys to ignore when comparing the stored configuration with the new configuration. Defaults to None.
        Raises:
            DuplicateNameError: If a configuration with the same name already exists.
        Examples:
            >>> store.store_trainer_config(trainer_config)
        """
        trainer_doc = converter.unstructure(trainer_config)
        self.__save_insert(self.trainers, trainer_doc, ignore)

    def retrieve_trainer_config(self, trainer_name):
        """
        Retrieve the trainer configuration for a given trainer name.

        Args:
            trainer_name (str): The name of the trainer configuration to retrieve.
        Returns:
            TrainerConfig: The trainer configuration object.
        Raises:
            KeyError: If the trainer name does not exist in the store.
        Examples:
            >>> trainer_config = store.retrieve_trainer_config("trainer1")

        """
        trainer_doc = self.__load(self.trainers, trainer_name)
        return converter.structure(trainer_doc, TrainerConfig)

    def retrieve_trainer_config_names(self):
        """
        Retrieve the names of the trainer configurations.

        Args:
            trainer_name (str): The name of the trainer configuration to retrieve.
        Returns:
            TrainerConfig: The trainer configuration object.
        Raises:
            KeyError: If the trainer name does not exist in the store.
        Examples:
            >>> trainer_config = store.retrieve_trainer_config("trainer1")
        """
        return [f.name[:-5] for f in self.trainers.iterdir()]

    def store_datasplit_config(self, datasplit_config, ignore=None):
        """
        Stores the datasplit configuration in the file config store.

        Args:
            datasplit_config (DataSplitConfig): The datasplit configuration to store.
            ignore (list, optional): A list of keys to ignore when comparing the stored configuration with the new configuration. Defaults to None.
        Raises:
            DuplicateNameError: If a configuration with the same name already exists.
        Examples:
            >>> store.store_datasplit_config(datasplit_config)
        """
        datasplit_doc = converter.unstructure(datasplit_config)
        self.__save_insert(self.datasplits, datasplit_doc, ignore)

    def retrieve_datasplit_config(self, datasplit_name):
        """
        Retrieve the datasplit configuration for a given datasplit name.

        Args:
            datasplit_name (str): The name of the datasplit configuration to retrieve.
        Returns:
            DataSplitConfig: The datasplit configuration object.
        Raises:
            KeyError: If the datasplit name does not exist in the store.
        Examples:
            >>> datasplit_config = store.retrieve_datasplit_config("datasplit1")

        """
        datasplit_doc = self.__load(self.datasplits, datasplit_name)
        return converter.structure(datasplit_doc, DataSplitConfig)

    def retrieve_datasplit_config_names(self):
        """
        Retrieve the names of the datasplit configurations.

        Args:
            datasplit_name (str): The name of the datasplit configuration to retrieve.
        Returns:
            DataSplitConfig: The datasplit configuration object.
        Raises:
            KeyError: If the datasplit name does not exist in the store.
        Examples:
            >>> datasplit_config = store.retrieve_datasplit_config("datasplit1")

        """
        return [f.name[:-5] for f in self.datasplits.iterdir()]

    def store_array_config(self, array_config, ignore=None):
        """
        Stores the array configuration in the file config store.

        Args:
            array_config (ArrayConfig): The array configuration to store.
            ignore (list, optional): A list of keys to ignore when comparing the stored configuration with the new configuration. Defaults to None.
        Raises:
            DuplicateNameError: If a configuration with the same name already exists.
        Examples:
            >>> store.store_array_config(array_config)

        """
        array_doc = converter.unstructure(array_config)
        self.__save_insert(self.arrays, array_doc, ignore)

    def retrieve_array_config(self, array_name):
        """
        Retrieve the array configuration for a given array name.

        Args:
            array_name (str): The name of the array configuration to retrieve.
        Returns:
            ArrayConfig: The array configuration object.
        Raises:
            KeyError: If the array name does not exist in the store.
        Examples:
            >>> array_config = store.retrieve_array_config("array1")
        """
        array_doc = self.__load(self.arrays, array_name)
        return converter.structure(array_doc, ArrayConfig)

    def retrieve_array_config_names(self):
        """
        Retrieve the names of the array configurations.

        Returns:
            A list of array configuration names.
        Raises:
            KeyError: If no array configurations are stored.
        Examples:
            >>> array_names = store.retrieve_array_config_names()

        """
        return [f.name[:-5] for f in self.arrays.iterdir()]

    def __save_insert(self, collection, data, ignore=None):
        """
        Saves the data to the collection.

        Args:
            collection (Path): The path to the collection.
            data (dict): The data to store.
            ignore (list, optional): A list of keys to ignore when comparing the stored configuration with the new configuration. Defaults to None.
        Raises:
            DuplicateNameError: If a configuration with the same name already exists.
        Examples:
            >>> store.__save_insert(collection, data)
        """
        name = data["name"]

        file_store = collection / f"{name}.yaml"
        if not file_store.exists():
            with file_store.open("w") as f:
                yaml.dump(dict(data), f)

        else:
            with file_store.open("r") as f:
                existing = yaml.full_load(f)

            if not self.__same_doc(existing, data, ignore):
                raise DuplicateNameError(
                    f"Data for {name} does not match already stored "
                    f"entry. Found\n\n{existing}\n\nin DB, but was "
                    f"given\n\n{data}"
                )

    def __load(self, collection, name):
        """
        Loads the data from the collection.

        Args:
            collection (Path): The path to the collection.
            name (str): The name of the data to load.
        Returns:
            The data from the collection.
        Raises:
            ValueError: If the config with the name does not exist in the collection.
        Examples:
            >>> store.__load(collection, name)
        """
        file_store = collection / f"{name}.yaml"
        if file_store.exists():
            with file_store.open("r") as f:
                return yaml.full_load(f)
        else:
            raise ValueError(f"No config with name: {name} in collection: {collection}")

    def __same_doc(self, a, b, ignore=None):
        """
        Compares two dictionaries for equality, ignoring certain keys.

        Args:
            a (dict): The first dictionary to compare.
            b (dict): The second dictionary to compare.
            ignore (list, optional): A list of keys to ignore. Defaults to None.
        Returns:
            bool: True if the dictionaries are equal, False otherwise.
        Raises:
            KeyError: If the keys do not match.
        Examples:
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

        return a == b

    def __init_db(self):
        """
        Initializes the FileConfigStore database.
        Adds the collections for the FileConfigStore.

        Raises:
            FileNotFoundError: If the collections do not exist.
        Examples:
            >>> store.__init_db()

        """
        # no indexing for filesystem
        # please only use this config store for debugging
        pass

    def __open_collections(self):
        """
        Opens the collections for the FileConfigStore.

        Raises:
            FileNotFoundError: If the collections do not exist.
        Examples:
            >>> store.__open_collections()
        """
        self.users.mkdir(exist_ok=True, parents=True)
        self.runs.mkdir(exist_ok=True, parents=True)
        self.tasks.mkdir(exist_ok=True, parents=True)
        self.datasplits.mkdir(exist_ok=True, parents=True)
        self.arrays.mkdir(exist_ok=True, parents=True)
        self.architectures.mkdir(exist_ok=True, parents=True)
        self.trainers.mkdir(exist_ok=True, parents=True)

    @property
    def users(self) -> Path:
        """
        Returns the path to the users directory.

        Returns:
            Path: The path to the users directory.
        Raises:
            FileNotFoundError: If the users directory does not exist.
        Examples:
            >>> store.users
            Path("path/to/configs/users")
        """
        return self.path / "users"

    @property
    def runs(self) -> Path:
        """
        Returns the path to the runs directory.

        Returns:
            Path: The path to the runs directory.
        Raises:
            FileNotFoundError: If the runs directory does not exist.
        Examples:
            >>> store.runs
            Path("path/to/configs/runs")
        """
        return self.path / "runs"

    @property
    def tasks(self) -> Path:
        """
        Returns the path to the tasks directory.

        Returns:
            Path: The path to the tasks directory.
        Raises:
            FileNotFoundError: If the tasks directory does not exist.
        Examples:
            >>> store.tasks
            Path("path/to/configs/tasks")
        """
        return self.path / "tasks"

    @property
    def datasplits(self) -> Path:
        """
        Returns the path to the datasplits directory.

        Returns:
            Path: The path to the datasplits directory.
        Raises:
            FileNotFoundError: If the datasplits directory does not exist.
        Examples:
            >>> store.datasplits
            Path("path/to/configs/datasplits")
        """
        return self.path / "datasplits"

    @property
    def arrays(self) -> Path:
        """
        Returns the path to the arrays directory.

        Returns:
            Path: The path to the arrays directory.
        Raises:
            FileNotFoundError: If the arrays directory does not exist.
        Examples:
            >>> store.arrays
            Path("path/to/configs/arrays")
        """
        return self.path / "arrays"

    @property
    def architectures(self) -> Path:
        """
        Returns the path to the architectures directory.

        Returns:
            Path: The path to the architectures directory.
        Raises:
            FileNotFoundError: If the architectures directory does not exist.
        Examples:
            >>> store.architectures
            Path("path/to/configs/architectures")
        """
        return self.path / "architectures"

    @property
    def trainers(self) -> Path:
        """
        Returns the path to the trainers directory.

        Returns:
            Path: The path to the trainers directory.
        Raises:
            FileNotFoundError: If the trainers directory does not exist.
        Examples:
            >>> store.trainers
            Path("path/to/configs/trainers")
        """
        return self.path / "trainers"

    @property
    def datasets(self) -> Path:
        """
        Returns the path to the datasets directory.

        Returns:
            Path: The path to the datasets directory.
        Raises:
            FileNotFoundError: If the datasets directory does not exist.
        Examples:
            >>> store.datasets
            Path("path/to/configs/datasets")

        """
        return self.path / "datasets"

    def delete_config(self, database: Path, config_name: str) -> None:
        """
        Deletes a configuration file from the specified database.

        Args:
            database (Path): The path to the database where the configuration file is stored.
            config_name (str): The name of the configuration file to be deleted.
        Raises:
            FileNotFoundError: If the configuration file does not exist.
        Examples:
            >>> store.delete_config(Path("path/to/configs"), "run1")

        """
        (database / f"{config_name}.yaml").unlink()

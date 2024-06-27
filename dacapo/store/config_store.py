from abc import ABC, abstractmethod
from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from dacapo.experiments.run_config import RunConfig
    from dacapo.experiments.tasks.task_config import TaskConfig
    from dacapo.experiments.architectures.architecture_config import ArchitectureConfig
    from dacapo.experiments.datasplits.datasplit_config import DataSplitConfig
    from dacapo.experiments.datasplits.datasets.arrays.array_config import ArrayConfig
    from dacapo.experiments.trainers.trainer_config import TrainerConfig


class DuplicateNameError(Exception):
    """
    Exception raised when trying to store a config with a name that already
    exists.

    Attributes:
        message (str): The error message.
    Methods:
        __str__: Return the error message.

    """

    pass


class ConfigStore(ABC):
    """
    Base class for configuration stores.

    Attributes:
        runs (Any): The runs store.
        datasplits (Any): The datasplits store.
        datasets (Any): The datasets store.
        arrays (Any): The arrays store.
        tasks (Any): The tasks store.
        trainers (Any): The trainers store.
        architectures (Any): The architectures store.
    Methods:
        delete_config: Delete a config from a store.
        store_run_config: Store a run config.
        retrieve_run_config: Retrieve a run config from a run name.
        retrieve_run_config_names: Retrieve all run config names.
        delete_run_config: Delete a run config.
        store_task_config: Store a task config.
        retrieve_task_config: Retrieve a task config from a task name.
        retrieve_task_config_names: Retrieve all task config names.
        delete_task_config: Delete a task config.
        store_architecture_config: Store a architecture config.
        retrieve_architecture_config: Retrieve a architecture config from a architecture name.
        retrieve_architecture_config_names: Retrieve all architecture config names.
        delete_architecture_config: Delete a architecture config.
        store_trainer_config: Store a trainer config.
        retrieve_trainer_config: Retrieve a trainer config from a trainer name.
        retrieve_trainer_config_names: Retrieve all trainer config names.
        delete_trainer_config: Delete a trainer config.
        store_datasplit_config: Store a datasplit config.
        retrieve_datasplit_config: Retrieve a datasplit config from a datasplit name.
        retrieve_datasplit_config_names: Retrieve all datasplit names.
        delete_datasplit_config: Delete a datasplit config.
        store_array_config: Store a array config.
        retrieve_array_config: Retrieve a array config from a array name.
        retrieve_array_config_names: Retrieve all array names.
        delete_array_config: Delete a array config.
    Note:
        This class is an abstract base class for configuration stores. It
        defines the interface for storing and retrieving configuration objects
        (e.g., run, task, architecture, trainer, datasplit, dataset, array
        configs). Concrete implementations of this class should define how
        these objects are stored and retrieved (e.g., in a database, in files).

    """

    runs: Any
    datasplits: Any
    datasets: Any
    arrays: Any
    tasks: Any
    trainers: Any
    architectures: Any

    @abstractmethod
    def delete_config(self, database, config_name: str) -> None:
        """
        Delete a config from a store.

        Args:
            database (Any): The store to delete the config from.
            config_name (str): The name of the config to delete.
        Raises:
            KeyError: If the config does not exist.
        Examples:
            >>> store.delete_config(store.runs, "run1")

        """
        pass

    @abstractmethod
    def store_run_config(self, run_config: "RunConfig") -> None:
        """
        Store a run config. This should also store the configs that are part
        of the run config (i.e., task, architecture, trainer, and dataset
        config).

        Args:
            run_config (RunConfig): The run config to store.
        Raises:
            DuplicateNameError: If a run config with the same name already
                exists.
        Examples:
            >>> store.store_run_config(run_config)

        """
        pass

    @abstractmethod
    def retrieve_run_config(self, run_name: str) -> "RunConfig":
        """
        Retrieve a run config from a run name.

        Args:
            run_name (str): The name of the run config to retrieve.
        Returns:
            RunConfig: The run config with the given name.
        Raises:
            KeyError: If the run config does not exist.
        Examples:
            >>> run_config = store.retrieve_run_config("run1")

        """
        pass

    @abstractmethod
    def retrieve_run_config_names(self) -> List[str]:
        """
        Retrieve all run config names.

        Returns:
            List[str]: The names of all run configs.
        Raises:
            KeyError: If no run configs exist.
        Examples:
            >>> run_names = store.retrieve_run_config_names()
        """
        pass

    def delete_run_config(self, run_name: str) -> None:
        """
        Delete a run config from the store.

        Args:
            run_name (str): The name of the run config to delete.
        Raises:
            KeyError: If the run config does not exist.
        Examples:
            >>> store.delete_run_config("run1")

        """
        self.delete_config(self.runs, run_name)

    @abstractmethod
    def store_task_config(self, task_config: "TaskConfig") -> None:
        """
        Store a task config.

        Args:
            task_config (TaskConfig): The task config to store.
        Raises:
            DuplicateNameError: If a task config with the same name already
                exists.
        Examples:
            >>> store.store_task_config(task_config)

        """
        pass

    @abstractmethod
    def retrieve_task_config(self, task_name: str) -> "TaskConfig":
        """
        Retrieve a task config from a task name.

        Args:
            task_name (str): The name of the task config to retrieve.
        Returns:
            TaskConfig: The task config with the given name.
        Raises:
            KeyError: If the task config does not exist.
        Examples:
            >>> task_config = store.retrieve_task_config("task1")

        """
        pass

    @abstractmethod
    def retrieve_task_config_names(self) -> List[str]:
        """
        Retrieve all task config names.

        Args:
            List[str]: The names of all task configs.
        Returns:
            List[str]: The names of all task configs.
        Raises:
            KeyError: If no task configs exist.
        Examples:
            >>> task_names = store.retrieve_task_config_names()

        """
        pass

    def delete_task_config(self, task_name: str) -> None:
        """
        Delete a task config from the store.

        Args:
            task_name (str): The name of the task config to delete.
        Raises:
            KeyError: If the task config does not exist.
        Examples:
            >>> store.delete_task_config("task1")

        """
        self.delete_config(self.tasks, task_name)

    @abstractmethod
    def store_architecture_config(
        self, architecture_config: "ArchitectureConfig"
    ) -> None:
        """
        Store a architecture config.

        Args:
            architecture_config (ArchitectureConfig): The architecture config
                to store.
        Raises:
            DuplicateNameError: If a architecture config with the same name
                already exists.
        Examples:
            >>> store.store_architecture_config(architecture_config)
        """
        pass

    @abstractmethod
    def retrieve_architecture_config(
        self, architecture_name: str
    ) -> "ArchitectureConfig":
        """
        Retrieve a architecture config from a architecture name.

        Args:
            architecture_name (str): The name of the architecture config to
                retrieve.
        Returns:
            ArchitectureConfig: The architecture config with the given name.
        Raises:
            KeyError: If the architecture config does not exist.
        Examples:
            >>> architecture_config = store.retrieve_architecture_config("architecture1")
        """
        pass

    @abstractmethod
    def retrieve_architecture_config_names(self) -> List[str]:
        """
        Retrieve all architecture config names.

        Args:
            List[str]: The names of all architecture configs.
        Returns:
            List[str]: The names of all architecture configs.
        Raises:
            KeyError: If no architecture configs exist.
        Examples:
            >>> architecture_names = store.retrieve_architecture_config_names()
        """
        pass

    def delete_architecture_config(self, architecture_name: str) -> None:
        """
        Delete a architecture config from the store.

        Args:
            architecture_name (str): The name of the architecture config to
                delete.
        Raises:
            KeyError: If the architecture config does not exist.
        Examples:
            >>> store.delete_architecture_config("architecture1")
        """
        self.delete_config(self.architectures, architecture_name)

    @abstractmethod
    def store_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        """
        Store a trainer config.

        Args:
            trainer_config (TrainerConfig): The trainer config to store.
        Raises:
            DuplicateNameError: If a trainer config with the same name already
                exists.
        Examples:
            >>> store.store_trainer_config(trainer_config)
        """
        pass

    @abstractmethod
    def retrieve_trainer_config(self, trainer_name: str) -> None:
        """
        Retrieve a trainer config from a trainer name.

        Args:
            trainer_name (str): The name of the trainer config to retrieve.
        Returns:
            TrainerConfig: The trainer config with the given name.
        Raises:
            KeyError: If the trainer config does not exist.
        Examples:
            >>> trainer_config = store.retrieve_trainer_config("trainer1")
        """
        pass

    @abstractmethod
    def retrieve_trainer_config_names(self) -> List[str]:
        """
        Retrieve all trainer config names.

        Args:
            List[str]: The names of all trainer configs.
        Returns:
            List[str]: The names of all trainer configs.
        Raises:
            KeyError: If no trainer configs exist.
        Examples:
            >>> trainer_names = store.retrieve_trainer_config_names()

        """
        pass

    def delete_trainer_config(self, trainer_name: str) -> None:
        """
        Delete a trainer config from the store.

        Args:
            trainer_name (str): The name of the trainer config to delete.
        Raises:
            KeyError: If the trainer config does not exist.
        Examples:
            >>> store.delete_trainer_config("trainer1")

        """
        self.delete_config(self.trainers, trainer_name)

    @abstractmethod
    def store_datasplit_config(self, datasplit_config: "DataSplitConfig") -> None:
        """
        Store a datasplit config.

        Args:
            datasplit_config (DataSplitConfig): The datasplit config to store.
        Raises:
            DuplicateNameError: If a datasplit config with the same name already
                exists.
        Examples:
            >>> store.store_datasplit_config(datasplit_config)
        """
        pass

    @abstractmethod
    def retrieve_datasplit_config(self, datasplit_name: str) -> "DataSplitConfig":
        """
        Retrieve a datasplit config from a datasplit name.

        Args:
            datasplit_name (str): The name of the datasplit config to retrieve.
        Returns:
            DataSplitConfig: The datasplit config with the given name.
        Raises:
            KeyError: If the datasplit config does not exist.
        Examples:
            >>> datasplit_config = store.retrieve_datasplit_config("datasplit1")
        """
        pass

    @abstractmethod
    def retrieve_datasplit_config_names(self) -> List[str]:
        """
        Retrieve all datasplit names.

        Args:
            List[str]: The names of all datasplit configs.
        Returns:
            List[str]: The names of all datasplit configs.
        Raises:
            KeyError: If no datasplit configs exist.
        Examples:
            >>> datasplit_names = store.retrieve_datasplit_config_names()

        """
        pass

    def delete_datasplit_config(self, datasplit_name: str) -> None:
        self.delete_config(self.datasplits, datasplit_name)

    @abstractmethod
    def store_array_config(self, array_config: "ArrayConfig") -> None:
        """
        Store a array config.

        Args:
            array_config (ArrayConfig): The array config to store.
        Raises:
            DuplicateNameError: If a array config with the same name already
                exists.
        Examples:
            >>> store.store_array_config(array_config)
        """
        pass

    @abstractmethod
    def retrieve_array_config(self, array_name: str) -> "ArrayConfig":
        """
        Retrieve a array config from a array name.

        Args:
            array_name (str): The name of the array config to retrieve.
        Returns:
            ArrayConfig: The array config with the given name.
        Raises:
            KeyError: If the array config does not exist.
        Examples:
            >>> array_config = store.retrieve_array_config("array1")
        """
        pass

    @abstractmethod
    def retrieve_array_config_names(self) -> List[str]:
        """
        Retrieve all array names.

        Args:
            List[str]: The names of all array configs.
        Returns:
            List[str]: The names of all array configs.
        Raises:
            KeyError: If no array configs exist.
        Examples:
            >>> array_names = store.retrieve_array_config_names()
        """
        pass

    def delete_array_config(self, array_name: str) -> None:
        """
        Delete a array config from the store.

        Args:
            array_name (str): The name of the array config to delete.
        Raises:
            KeyError: If the array config does not exist.
        Examples:
            >>> store.delete_array_config("array1")
        """
        self.delete_config(self.arrays, array_name)

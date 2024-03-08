from abc import ABC, abstractmethod
from typing import Any, List, TYPE_CHECKING
from .converter import converter

if TYPE_CHECKING:
    from dacapo.experiments.run_config import RunConfig
    from dacapo.experiments.tasks.task_config import TaskConfig
    from dacapo.experiments.architectures.architecture_config import ArchitectureConfig
    from dacapo.experiments.datasplits.datasplit_config import DataSplitConfig
    from dacapo.experiments.datasplits.datasets.arrays.array_config import ArrayConfig
    from dacapo.experiments.trainers.trainer_config import TrainerConfig


class DuplicateNameError(Exception):
    pass


class ConfigStore(ABC):
    """Base class for configuration stores."""

    runs: Any
    datasplits: Any
    datasets: Any
    arrays: Any
    tasks: Any
    trainers: Any
    architectures: Any

    @abstractmethod
    def delete_config(self, database, config_name: str) -> None:
        pass

    @abstractmethod
    def __save_insert(self, collection, data, ignore=None):
        pass

    def store_run_config(self, run_config: "RunConfig") -> None:
        """Store a run config. This should also store the configs that are part
        of the run config (i.e., task, architecture, trainer, and dataset
        config)."""
        run_doc = converter.unstructure(run_config)
        self.__save_insert(self.runs, run_doc)

    @abstractmethod
    def retrieve_run_config(self, run_name: str) -> "RunConfig":
        """Retrieve a run config from a run name."""
        pass

    @abstractmethod
    def retrieve_run_config_names(self) -> List[str]:
        """Retrieve all run config names."""
        pass

    def delete_run_config(self, run_name: str) -> None:
        self.delete_config(self.runs, run_name)

    def store_task_config(self, task_config: "TaskConfig") -> None:
        """Store a task config."""
        task_doc = converter.unstructure(task_config)
        self.__save_insert(self.tasks, task_doc)

    @abstractmethod
    def retrieve_task_config(self, task_name: str) -> "TaskConfig":
        """Retrieve a task config from a task name."""
        pass

    @abstractmethod
    def retrieve_task_config_names(self) -> List[str]:
        """Retrieve all task config names."""
        pass

    def delete_task_config(self, task_name: str) -> None:
        self.delete_config(self.tasks, task_name)

    def store_architecture_config(
        self, architecture_config: "ArchitectureConfig"
    ) -> None:
        """Store a architecture config."""
        architecture_doc = converter.unstructure(architecture_config)
        self.__save_insert(self.architectures, architecture_doc)

    @abstractmethod
    def retrieve_architecture_config(
        self, architecture_name: str
    ) -> "ArchitectureConfig":
        """Retrieve a architecture config from a architecture name."""
        pass

    @abstractmethod
    def retrieve_architecture_config_names(self) -> List[str]:
        """Retrieve all architecture config names."""
        pass

    def delete_architecture_config(self, architecture_name: str) -> None:
        self.delete_config(self.architectures, architecture_name)

    def store_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        """Store a trainer config."""
        trainer_doc = converter.unstructure(trainer_config)
        self.__save_insert(self.trainers, trainer_doc)

    @abstractmethod
    def retrieve_trainer_config(self, trainer_name: str) -> None:
        """Retrieve a trainer config from a trainer name."""
        pass

    @abstractmethod
    def retrieve_trainer_config_names(self) -> List[str]:
        """Retrieve all trainer config names."""
        pass

    def delete_trainer_config(self, trainer_name: str) -> None:
        self.delete_config(self.trainers, trainer_name)

    def store_datasplit_config(self, datasplit_config: "DataSplitConfig") -> None:
        """Store a datasplit config."""
        datasplit_doc = converter.unstructure(datasplit_config)
        self.__save_insert(self.datasplits, datasplit_doc)

    @abstractmethod
    def retrieve_datasplit_config(self, datasplit_name: str) -> "DataSplitConfig":
        """Retrieve a datasplit config from a datasplit name."""
        pass

    @abstractmethod
    def retrieve_datasplit_config_names(self) -> List[str]:
        """Retrieve all datasplit names."""
        pass

    def delete_datasplit_config(self, datasplit_name: str) -> None:
        self.delete_config(self.datasplits, datasplit_name)

    def store_array_config(self, array_config: "ArrayConfig") -> None:
        """Store a array config."""
        array_doc = converter.unstructure(array_config)
        self.__save_insert(self.arrays, array_doc)

    @abstractmethod
    def retrieve_array_config(self, array_name: str) -> "ArrayConfig":
        """Retrieve a array config from a array name."""
        pass

    @abstractmethod
    def retrieve_array_config_names(self) -> List[str]:
        """Retrieve all array names."""
        pass

    def delete_array_config(self, array_name: str) -> None:
        self.delete_config(self.arrays, array_name)

    def store_dataset_config(self, dataset_config):
        dataset_doc = converter.unstructure(dataset_config)
        self.__save_insert(self.datasets, dataset_doc)

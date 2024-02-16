from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

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

    @property
    @abstractmethod
    def runs(self):
        pass

    @property
    @abstractmethod
    def datasplits(self):
        pass

    @property
    @abstractmethod
    def datasets(self):
        pass

    @property
    @abstractmethod
    def arrays(self):
        pass

    @property
    @abstractmethod
    def tasks(self):
        pass

    @property
    @abstractmethod
    def trainers(self):
        pass

    @property
    @abstractmethod
    def architectures(self):
        pass

    @abstractmethod
    def delete_config(self, database, config_name: str) -> None:
        pass

    @abstractmethod
    def store_run_config(self, run_config: "RunConfig") -> None:
        """Store a run config. This should also store the configs that are part
        of the run config (i.e., task, architecture, trainer, and dataset
        config)."""

    @abstractmethod
    def retrieve_run_config(self, run_name: str) -> "RunConfig":
        """Retrieve a run config from a run name."""

    @abstractmethod
    def retrieve_run_config_names(self) -> List[str]:
        """Retrieve all run config names."""

    def delete_run_config(self, run_name: str) -> None:
        self.delete_config(self.runs, run_name)

    @abstractmethod
    def store_task_config(self, task_config: "TaskConfig") -> None:
        """Store a task config."""

    @abstractmethod
    def retrieve_task_config(self, task_name: str) -> "TaskConfig":
        """Retrieve a task config from a task name."""

    @abstractmethod
    def retrieve_task_config_names(self) -> List[str]:
        """Retrieve all task config names."""

    def delete_task_config(self, task_name: str) -> None:
        self.delete_config(self.tasks, task_name)

    @abstractmethod
    def store_architecture_config(
        self, architecture_config: "ArchitectureConfig"
    ) -> None:
        """Store a architecture config."""

    @abstractmethod
    def retrieve_architecture_config(
        self, architecture_name: str
    ) -> "ArchitectureConfig":
        """Retrieve a architecture config from a architecture name."""

    @abstractmethod
    def retrieve_architecture_config_names(self) -> List[str]:
        """Retrieve all architecture config names."""

    def delete_architecture_config(self, architecture_name: str) -> None:
        self.delete_config(self.architectures, architecture_name)

    @abstractmethod
    def store_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        """Store a trainer config."""

    @abstractmethod
    def retrieve_trainer_config(self, trainer_name: str) -> None:
        """Retrieve a trainer config from a trainer name."""

    @abstractmethod
    def retrieve_trainer_config_names(self) -> List[str]:
        """Retrieve all trainer config names."""

    def delete_trainer_config(self, trainer_name: str) -> None:
        self.delete_config(self.trainers, trainer_name)

    @abstractmethod
    def store_datasplit_config(self, datasplit_config: "DataSplitConfig") -> None:
        """Store a datasplit config."""

    @abstractmethod
    def retrieve_datasplit_config(self, datasplit_name: str) -> "DataSplitConfig":
        """Retrieve a datasplit config from a datasplit name."""

    @abstractmethod
    def retrieve_datasplit_config_names(self) -> List[str]:
        """Retrieve all datasplit names."""

    def delete_datasplit_config(self, datasplit_name: str) -> None:
        self.delete_config(self.datasplits, datasplit_name)

    @abstractmethod
    def store_array_config(self, array_config: "ArrayConfig") -> None:
        """Store a array config."""

    @abstractmethod
    def retrieve_array_config(self, array_name: str) -> "ArrayConfig":
        """Retrieve a array config from a array name."""

    @abstractmethod
    def retrieve_array_config_names(self) -> List[str]:
        """Retrieve all array names."""

    def delete_array_config(self, array_name: str) -> None:
        self.delete_config(self.arrays, array_name)

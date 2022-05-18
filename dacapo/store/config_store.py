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

    @abstractmethod
    def store_run_config(self, run_config: "RunConfig") -> None:
        """Store a run config. This should also store the configs that are part
        of the run config (i.e., task, architecture, trainer, and dataset
        config)."""
        pass

    @abstractmethod
    def retrieve_run_config(self, run_name: str) -> "RunConfig":
        """Retrieve a run config from a run name."""
        pass

    @abstractmethod
    def retrieve_run_config_names(self) -> List[str]:
        """Retrieve all run config names."""
        pass

    @abstractmethod
    def store_task_config(self, task_config: "TaskConfig") -> None:
        """Store a task config."""
        pass

    @abstractmethod
    def retrieve_task_config(self, task_name: str) -> "TaskConfig":
        """Retrieve a task config from a task name."""
        pass

    @abstractmethod
    def retrieve_task_config_names(self) -> List[str]:
        """Retrieve all task config names."""
        pass

    @abstractmethod
    def store_architecture_config(
        self, architecture_config: "ArchitectureConfig"
    ) -> None:
        """Store a architecture config."""
        pass

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

    @abstractmethod
    def store_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        """Store a trainer config."""
        pass

    @abstractmethod
    def retrieve_trainer_config(self, trainer_name: str) -> None:
        """Retrieve a trainer config from a trainer name."""
        pass

    @abstractmethod
    def retrieve_trainer_config_names(self) -> List[str]:
        """Retrieve all trainer config names."""
        pass

    @abstractmethod
    def store_datasplit_config(self, datasplit_config: "DataSplitConfig") -> None:
        """Store a datasplit config."""
        pass

    @abstractmethod
    def retrieve_datasplit_config(self, datasplit_name: str) -> "DataSplitConfig":
        """Retrieve a datasplit config from a datasplit name."""
        pass

    @abstractmethod
    def retrieve_datasplit_config_names(self) -> List[str]:
        """Retrieve all datasplit names."""
        pass

    @abstractmethod
    def store_array_config(self, array_config: "ArrayConfig") -> None:
        """Store a array config."""
        pass

    @abstractmethod
    def retrieve_array_config(self, array_name: str) -> "ArrayConfig":
        """Retrieve a array config from a array name."""
        pass

    @abstractmethod
    def retrieve_array_config_names(self) -> List[str]:
        """Retrieve all array names."""
        pass

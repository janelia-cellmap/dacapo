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
    

    pass


class ConfigStore(ABC):
    

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
    def store_run_config(self, run_config: "RunConfig") -> None:
        
        pass

    @abstractmethod
    def retrieve_run_config(self, run_name: str) -> "RunConfig":
        
        pass

    @abstractmethod
    def retrieve_run_config_names(self) -> List[str]:
        
        pass

    def delete_run_config(self, run_name: str) -> None:
        
        self.delete_config(self.runs, run_name)

    @abstractmethod
    def store_task_config(self, task_config: "TaskConfig") -> None:
        
        pass

    @abstractmethod
    def retrieve_task_config(self, task_name: str) -> "TaskConfig":
        
        pass

    @abstractmethod
    def retrieve_task_config_names(self) -> List[str]:
        
        pass

    def delete_task_config(self, task_name: str) -> None:
        
        self.delete_config(self.tasks, task_name)

    @abstractmethod
    def store_architecture_config(
        self, architecture_config: "ArchitectureConfig"
    ) -> None:
        
        pass

    @abstractmethod
    def retrieve_architecture_config(
        self, architecture_name: str
    ) -> "ArchitectureConfig":
        
        pass

    @abstractmethod
    def retrieve_architecture_config_names(self) -> List[str]:
        
        pass

    def delete_architecture_config(self, architecture_name: str) -> None:
        
        self.delete_config(self.architectures, architecture_name)

    @abstractmethod
    def store_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        
        pass

    @abstractmethod
    def retrieve_trainer_config(self, trainer_name: str) -> None:
        
        pass

    @abstractmethod
    def retrieve_trainer_config_names(self) -> List[str]:
        
        pass

    def delete_trainer_config(self, trainer_name: str) -> None:
        
        self.delete_config(self.trainers, trainer_name)

    @abstractmethod
    def store_datasplit_config(self, datasplit_config: "DataSplitConfig") -> None:
        
        pass

    @abstractmethod
    def retrieve_datasplit_config(self, datasplit_name: str) -> "DataSplitConfig":
        
        pass

    @abstractmethod
    def retrieve_datasplit_config_names(self) -> List[str]:
        
        pass

    def delete_datasplit_config(self, datasplit_name: str) -> None:
        self.delete_config(self.datasplits, datasplit_name)

    @abstractmethod
    def store_array_config(self, array_config: "ArrayConfig") -> None:
        
        pass

    @abstractmethod
    def retrieve_array_config(self, array_name: str) -> "ArrayConfig":
        
        pass

    @abstractmethod
    def retrieve_array_config_names(self) -> List[str]:
        
        pass

    def delete_array_config(self, array_name: str) -> None:
        
        self.delete_config(self.arrays, array_name)

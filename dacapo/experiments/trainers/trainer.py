import torch

from abc import ABC, abstractmethod
from typing import List, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from dacapo.experiments.model import Model
    from dacapo.experiments.datasplits.datasets import Dataset
    from dacapo.experiments.tasks.task import Task
    from dacapo.experiments.training_iteration_stats import TrainingIterationStats
    from dacapo.store.array_store import LocalContainerIdentifier


class Trainer(ABC):
    iteration: int
    batch_size: int
    learning_rate: float

    @abstractmethod
    def create_optimizer(self, model: "Model") -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def iterate(
        self,
        num_iterations: int,
        model: "Model",
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Iterator["TrainingIterationStats"]:
        pass

    @abstractmethod
    def can_train(self, datasets: List["Dataset"]) -> bool:
        pass

    @abstractmethod
    def build_batch_provider(
        self,
        datasets: List["Dataset"],
        model: "Model",
        task: "Task",
        snapshot_container: "LocalContainerIdentifier",
    ) -> None:
        pass

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

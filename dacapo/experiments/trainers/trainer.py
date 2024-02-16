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
        """Create a ``torch`` optimizer for the given model."""

    @abstractmethod
    def iterate(
        self,
        num_iterations: int,
        model: "Model",
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Iterator["TrainingIterationStats"]:
        """Perform ``num_iterations`` training iterations."""

    @abstractmethod
    def can_train(self, datasets: List["Dataset"]) -> bool:
        """
        Can this trainer train with a specific set of datasets. Some trainers
        may have requirements for their training datasets.
        """

    @abstractmethod
    def build_batch_provider(
        self,
        datasets: List["Dataset"],
        model: "Model",
        task: "Task",
        snapshot_container: "LocalContainerIdentifier",
    ) -> None:
        """
        Initialize the training pipeline using the datasets, model, task
        and snapshot_container

        The training datasets are required s.t. the pipeline knows where to pull
        data from.
        The model is needed to inform the pipeline of required input/output sizes
        The task is needed to transform gt into target
        The snapshot_container defines where snapshots will be saved.
        """

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

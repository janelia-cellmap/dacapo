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
    """
    Trainer Abstract Base Class

    This serves as the blueprint for any trainer classes in the dacapo library.
    It defines essential methods that every subclass must implement for effective
    training of a neural network model.

    Attributes:
        iteration (int): The number of training iterations.
        batch_size (int): The size of the training batch.
        learning_rate (float): The learning rate for the optimizer.
    Methods:
        create_optimizer(model: Model) -> torch.optim.Optimizer:
            Creates an optimizer for the model.
        iterate(num_iterations: int, model: Model, optimizer: torch.optim.Optimizer, device: torch.device) -> Iterator[TrainingIterationStats]:
            Performs a number of training iterations.
        can_train(datasets: List[Dataset]) -> bool:
            Checks if the trainer can train with a specific set of datasets.
        build_batch_provider(datasets: List[Dataset], model: Model, task: Task, snapshot_container: LocalContainerIdentifier) -> None:
            Initializes the training pipeline using various components.
    Note:
        The Trainer class is an abstract class that cannot be instantiated directly. It is meant to be subclassed.
    """

    iteration: int
    batch_size: int
    learning_rate: float

    @abstractmethod
    def create_optimizer(self, model: "Model") -> torch.optim.Optimizer:
        """
        Creates an optimizer for the model.

        Args:
            model (Model): The model for which the optimizer will be created.
        Returns:
            torch.optim.Optimizer: The optimizer created for the model.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> optimizer = trainer.create_optimizer(model)
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def iterate(
        self,
        num_iterations: int,
        model: "Model",
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Iterator["TrainingIterationStats"]:
        """
        Performs a number of training iterations.

        Args:
            num_iterations (int): Number of training iterations.
            model (Model): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            device (torch.device): The device (GPU/CPU) where the model will be trained.
        Returns:
            Iterator[TrainingIterationStats]: An iterator of the training statistics.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> for iteration_stats in trainer.iterate(num_iterations, model, optimizer, device):
            >>>     print(iteration_stats)
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def can_train(self, datasets: List["Dataset"]) -> bool:
        """
        Checks if the trainer can train with a specific set of datasets.

        Some trainers may have specific requirements for their training datasets.

        Args:
            datasets (List[Dataset]): The training datasets.
        Returns:
            bool: True if the trainer can train on the given datasets, False otherwise.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> can_train = trainer.can_train(datasets)
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def build_batch_provider(
        self,
        datasets: List["Dataset"],
        model: "Model",
        task: "Task",
        snapshot_container: "LocalContainerIdentifier",
    ) -> None:
        """Initializes the training pipeline using various components.

        This method uses the datasets, model, task, and snapshot_container to set up the
        training pipeline.

        Args:
            datasets (List[Dataset]): The datasets to pull data from.
            model (Model): The model to inform the pipeline of required input/output sizes.
            task (Task): The task to transform ground truth into target.
            snapshot_container (LocalContainerIdentifier): Defines where snapshots will be saved.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> trainer.build_batch_provider(datasets, model, task, snapshot_container)
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def __enter__(self):
        """
        Defines the functionality of the '__enter__' method for use in a 'with' statement.

        Returns:
            Trainer: The trainer object.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> with trainer as t:
            >>>     print(t)

        """
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Defines the functionality of the '__exit__' method for use in a 'with' statement.

        Args:
            exc_type: The type of exception raised.
            exc_val: The exception value.
            exc_tb: The traceback.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> with trainer as t:
            >>>     print(t)
        """
        pass

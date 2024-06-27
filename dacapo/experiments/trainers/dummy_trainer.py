from ..training_iteration_stats import TrainingIterationStats
from .trainer import Trainer
from dacapo.experiments.model import Model

import torch
import numpy as np


class DummyTrainer(Trainer):
    """
    This class is used to train a model using dummy data and is used for testing purposes. It contains attributes
    related to learning rate, batch size, and mirror augment. It also contains methods to create an optimizer, iterate
    over the training data, build a batch provider, and check if the trainer can train on the given data split. This class
    contains methods to enter and exit the context manager. The iterate method yields training iteration statistics.

    Attributes:
        learning_rate (float): The learning rate to use.
        batch_size (int): The batch size to use.
        mirror_augment (bool): A boolean value indicating whether to use mirror augmentation or not.
    Methods:
        __init__(self, trainer_config): This method initializes the DummyTrainer object.
        create_optimizer(self, model): This method creates an optimizer for the given model.
        iterate(self, num_iterations: int, model, optimizer, device): This method iterates over the training data for the specified number of iterations.
        build_batch_provider(self, datasplit, architecture, task, snapshot_container): This method builds a batch provider for the given data split, architecture, task, and snapshot container.
        can_train(self, datasplit): This method checks if the trainer can train on the given data split.
        __enter__(self): This method enters the context manager.
        __exit__(self, exc_type, exc_val, exc_tb): This method exits the context manager.
    Note:
        The iterate method yields TrainingIterationStats.
    """

    iteration = 0

    def __init__(self, trainer_config):
        """
        Initialize the DummyTrainer object.

        Args:
            trainer_config (TrainerConfig): The configuration object for the trainer.
        Returns:
            DummyTrainer: The DummyTrainer object.
        Examples:
            >>> trainer = DummyTrainer(trainer_config)

        """
        self.learning_rate = trainer_config.learning_rate
        self.batch_size = trainer_config.batch_size
        self.mirror_augment = trainer_config.mirror_augment

    def create_optimizer(self, model):
        """
        Create an optimizer for the given model.

        Args:
            model (Model): The model to optimize.
        Returns:
            torch.optim.Optimizer: The optimizer object.
        Examples:
            >>> optimizer = create_optimizer(model)

        """
        return torch.optim.RAdam(lr=self.learning_rate, params=model.parameters())

    def iterate(self, num_iterations: int, model: Model, optimizer, device):
        """
        Iterate over the training data for the specified number of iterations.

        Args:
            num_iterations (int): The number of iterations to perform.
            model (Model): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            device (torch.device): The device to perform the computations on.
        Yields:
            TrainingIterationStats: The training iteration statistics.
        Raises:
            ValueError: If the number of iterations is less than or equal to zero.
        Examples:
            >>> for stats in iterate(num_iterations, model, optimizer, device):
            >>>     print(stats)
        """
        target_iteration = self.iteration + num_iterations

        for iteration in range(self.iteration, target_iteration):
            optimizer.zero_grad()
            raw = (
                torch.from_numpy(
                    np.random.randn(1, model.num_in_channels, *model.input_shape)
                )
                .float()
                .to(device)
            )
            target = (
                torch.from_numpy(
                    np.zeros((1, model.num_out_channels, *model.output_shape))
                )
                .float()
                .to(device)
            )
            pred = model.forward(raw)
            loss = self._loss.compute(pred, target)
            loss.backward()
            optimizer.step()
            yield TrainingIterationStats(
                loss=1.0 / (iteration + 1), iteration=iteration, time=0.1
            )
            self.iteration += 1

    def build_batch_provider(self, datasplit, architecture, task, snapshot_container):
        """
        Build a batch provider for the given data split, architecture, task, and snapshot container.

        Args:
            datasplit (DataSplit): The data split to use.
            architecture (Architecture): The architecture to use.
            task (Task): The task to perform.
            snapshot_container (SnapshotContainer): The snapshot container to use.
        Returns:
            BatchProvider: The batch provider object.
        Raises:
            ValueError: If the task loss is not set.
        Examples:
            >>> batch_provider = build_batch_provider(datasplit, architecture, task, snapshot_container)

        """
        self._loss = task.loss

    def can_train(self, datasplit):
        """
        Check if the trainer can train on the given data split.

        Args:
            datasplit (DataSplit): The data split to check.
        Returns:
            bool: True if the trainer can train on the data split, False otherwise.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> can_train(datasplit)

        """
        return True

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            DummyTrainer: The trainer object.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> with trainer as t:
            >>>     print(t)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.

        Args:
            exc_type: The type of the exception.
            exc_val: The exception value.
            exc_tb: The exception traceback.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> with trainer as t:
            >>>     print(t)
        """
        pass

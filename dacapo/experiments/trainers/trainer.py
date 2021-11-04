from abc import ABC, abstractmethod
from contextlib import contextmanager


class Trainer(ABC):
    def __init__(self, trainer_config):

        self.learning_rate = trainer_config.learning_rate
        self.batch_size = trainer_config.batch_size
        self.iteration = 0

    def set_iteration(self, iteration):
        """Set the iteration for this trainer when resuming training."""

        self.iteration = iteration

    @abstractmethod
    def create_optimizer(self, model):
        """Create a ``torch`` optimizer for the given model."""
        pass

    @abstractmethod
    def iterate(self, num_iterations, model, optimizer):
        """Perform ``num_iterations`` training iterations. Each iteration
        should ``yield`` an instance of ``TrainingIterationStats``."""
        pass

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

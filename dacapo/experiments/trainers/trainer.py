from abc import ABC, abstractmethod
from contextlib import contextmanager


class Trainer(ABC):
    @property
    @abstractmethod
    def learning_rate(self):
        pass

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @property
    @abstractmethod
    def iteration(self):
        pass

    def set_iteration(self, iteration):
        """Set the iteration for this trainer when resuming training."""

        self.iteration = iteration

    @abstractmethod
    def create_optimizer(self, model):
        """Create a ``torch`` optimizer for the given model."""
        pass

    @abstractmethod
    def iterate(self, num_iterations, model, optimizer, device):
        """Perform ``num_iterations`` training iterations. Each iteration
        should ``yield`` an instance of ``TrainingIterationStats``."""
        pass

    @abstractmethod
    def can_train(self, datasets):
        """
        Can this trainer train with a specific set of datasets. Some trainers
        may have requirements for their training datasets.
        """
        pass

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

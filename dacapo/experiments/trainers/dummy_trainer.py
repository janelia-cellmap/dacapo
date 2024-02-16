"""
This module contains the class `DummyTrainer` that inherits from the base class `Trainer`.
It is used for training with a specified configurations and optimizer. The primary functions in 
this class include creating an optimizer, running training iterations, building batch providers, 
and conducting a training ability check. 
"""
from ..training_iteration_stats import TrainingIterationStats
from .trainer import Trainer
from dacapo.experiments.model import Model

import torch
import numpy as np


class DummyTrainer(Trainer):
    """
    The DummyTrainer class inherits from the `Trainer` and implements and overrides several
    functions such as `create_optimizer`,`iterate`,`build_batch_provider`,`can_train`, `__enter__` and `__exit__`
    """
    iteration = 0

    def __init__(self, trainer_config):
        """
        Instantiates a new object of this class with a trainer configuration.

        Args:
            trainer_config : The configuration parameters for the trainer.
        """
        self.learning_rate = trainer_config.learning_rate
        self.batch_size = trainer_config.batch_size
        self.mirror_augment = trainer_config.mirror_augment

    def create_optimizer(self, model):
        """
        Creates and returns an optimizer for the model.

        Args:
            model : The model for which the optimizer is to be created.

        Returns:
            Optimizer for the model.
        """
        return torch.optim.Adam(lr=self.learning_rate, params=model.parameters())

    def iterate(self, num_iterations: int, model: Model, optimizer, device):
        """
        Runs training iterations for a given number of iterations.

        Args:
            num_iterations (int): The number of training iterations to be run.
            model (Model): The model to be trained.
            optimizer : Optimizer used for training the model.
            device : Device to be used for training (gpu or cpu).
        """
        target_iteration = self.iteration + num_iterations
        ...
        
    def build_batch_provider(self, datasplit, architecture, task, snapshot_container):
        """
        Builds a batch provider.

        Args:
            datasplit : Data to be used for training.
            architecture: The model's architecture.
            task: The task for which the model is being trained.
            snapshot_container: The container for snapshots of training process.
        """
        self._loss = task.loss

    def can_train(self, datasplit):
        """
        Checks whether the training can be conducted.
  
        Args:
            datasplit: Data to be used for training.

        Returns:
            boolean: The return value. True for trainable, False otherwise.
        """
        return True
        
    def __enter__(self):
        """
        Manages the context behaviour during the enter phase of context management protocol.

        Returns:
            itself: An instance of the same class.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Manages the context behaviour during the exit phase of context management protocol.

        Args:
            exc_type: The type of exception.
            exc_value: The exception instance.
            traceback: A traceback object encapsulating the call stack.
        """
        pass
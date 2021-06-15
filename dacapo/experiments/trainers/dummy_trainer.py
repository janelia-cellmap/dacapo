import torch

from .trainer import Trainer
from .training_iteration_stats import TrainingIterationStats


class DummyTrainer(Trainer):

    def __init__(self, trainer_config):

        super().__init__(trainer_config)

        self.mirror_augment = trainer_config.mirror_augment

    def create_optimizer(self, model):

        return torch.optim.Adam(
            lr=self.learning_rate,
            params=model.parameters())

    def iterate(self, num_iterations):

        target_iteration = self.iteration + num_iterations

        for self.iteration in range(self.iteration, target_iteration):
            yield TrainingIterationStats(
                loss=1.0/(self.iteration + 1),
                iteration=self.iteration)

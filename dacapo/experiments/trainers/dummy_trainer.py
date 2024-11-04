from ..training_iteration_stats import TrainingIterationStats
from .trainer import Trainer
from dacapo.experiments.model import Model

import torch
import numpy as np


class DummyTrainer(Trainer):
    iteration = 0

    def __init__(self, trainer_config):
        self.learning_rate = trainer_config.learning_rate
        self.batch_size = trainer_config.batch_size
        self.mirror_augment = trainer_config.mirror_augment

    def create_optimizer(self, model):
        return torch.optim.RAdam(lr=self.learning_rate, params=model.parameters())

    def iterate(self, num_iterations: int, model: Model, optimizer, device):
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
        self._loss = task.loss

    def can_train(self, datasplit):
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

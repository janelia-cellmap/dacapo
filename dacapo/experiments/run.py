from .datasplits import DataSplit
from .training_stats import TrainingStats
from .validation_scores import ValidationScores

import torch


class Run:
    def __init__(self, run_config):

        self.name = run_config.name

        task_type = run_config.task_config.task_type
        architecture_type = run_config.architecture_config.architecture_type
        trainer_type = run_config.trainer_config.trainer_type
        datasplit_type = run_config.datasplit_config.datasplit_type

        self.task = task_type(run_config.task_config)
        self.architecture = architecture_type(run_config.architecture_config)
        self.trainer = trainer_type(run_config.trainer_config)
        self.datasplit = datasplit_type(run_config.datasplit_config)

        self.model = self.task.create_model(self.architecture)
        self.optimizer = self.trainer.create_optimizer(self.model)

        self.training_stats = TrainingStats()
        self.validation_scores = ValidationScores(
            self.task.parameters, self.datasplit.validate, self.task.evaluation_scores
        )

    def move_optimizer(self, device, empty_cuda_cache=False):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        if empty_cuda_cache:
            torch.cuda.empty_cache()

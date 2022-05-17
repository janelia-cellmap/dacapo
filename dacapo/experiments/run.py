from .datasplits import DataSplit
from .training_stats import TrainingStats
from .validation_scores import ValidationScores
from .starts import Start

import torch


class Run:
    def __init__(self, run_config):

        self.name = run_config.name

        # config types
        task_type = run_config.task_config.task_type
        architecture_type = run_config.architecture_config.architecture_type
        trainer_type = run_config.trainer_config.trainer_type
        datasplit_type = run_config.datasplit_config.datasplit_type

        # run components
        self.task = task_type(run_config.task_config)
        self.architecture = architecture_type(run_config.architecture_config)
        self.trainer = trainer_type(run_config.trainer_config)
        self.datasplit = datasplit_type(run_config.datasplit_config)

        # combined pieces
        self.model = self.task.create_model(self.architecture)
        self.optimizer = self.trainer.create_optimizer(self.model)

        # tracking
        self.training_stats = TrainingStats()
        self.validation_scores = ValidationScores(
            self.task.parameters, self.datasplit.validate, self.task.evaluation_scores
        )

        # preloaded weights from previous run
        self.start = (
            Start(run_config.start_config)
            if run_config.start_config is not None
            else None
        )
        if self.start is not None:
            self.start.initialize_weights(self.model)

    @staticmethod
    def get_validation_scores(run_config) -> None:
        """
        Static method to avoid having to initialize model, optimizer, trainer, etc.
        """
        task_type = run_config.task_config.task_type
        datasplit_type = run_config.datasplit_config.datasplit_type

        task = task_type(run_config.task_config)
        datasplit = datasplit_type(run_config.datasplit_config)

        return ValidationScores(
            task.parameters, datasplit.validate, task.evaluation_scores
        )

    def move_optimizer(self, device: torch.Device, empty_cuda_cache: bool = False) -> None:
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        if empty_cuda_cache:
            torch.cuda.empty_cache()

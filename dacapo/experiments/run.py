from .datasets import Dataset
from .training_stats import TrainingStats
from .validation_scores import ValidationScores


class Run:

    def __init__(self, run_config):

        self.name = run_config.name

        task_type = run_config.task_config.task_type
        architecture_type = run_config.architecture_config.architecture_type
        trainer_type = run_config.trainer_config.trainer_type

        self.task = task_type(run_config.task_config)
        self.architecture = architecture_type(run_config.architecture_config)
        self.trainer = trainer_type(run_config.trainer_config)
        self.dataset = Dataset(run_config.dataset_config)

        self.model = self.task.predictor.create_model(
            self.architecture,
            self.dataset)
        self.optimizer = self.trainer.create_optimizer(self.model)

        self.training_stats = TrainingStats()
        self.validation_scores = ValidationScores()

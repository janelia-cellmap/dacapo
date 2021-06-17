from .datasets import Dataset
from .training_stats import TrainingStats
from .validation_scores import ValidationScores


class Run:

    def __init__(self, run_config):

        task_type = run_config.task_config.task_type
        architecture_type = run_config.architecture_config.architecture_type
        trainer_type = run_config.trainer_config.trainer_type

        self.name = run_config.name

        self.task = task_type(run_config.task_config)
        self.architecture = architecture_type(run_config.architecture_config)
        self.trainer = trainer_type(run_config.trainer_config)
        self.dataset = Dataset(run_config.dataset_config)

        self.training_stats = TrainingStats()
        self.validation_scores = ValidationScores()

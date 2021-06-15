import attr

from .architectures import ArchitectureConfig
from .datasets import DatasetConfig
from .tasks import TaskConfig
from .trainers import TrainerConfig


@attr.s
class RunConfig:

    task_config: TaskConfig = attr.ib()
    architecture_config: ArchitectureConfig = attr.ib()
    trainer_config: TrainerConfig = attr.ib()
    dataset_config: DatasetConfig = attr.ib()

    repetition: int = attr.ib()
    num_iterations: int = attr.ib()

    validation_score: str = attr.ib(
        metadata={
            'help_text':
                "The name of the score used to compare validation results."
        })
    validation_score_minimize: bool = attr.ib(
        default=True,
        metadata={
            'help_text':
                "Whether lower validation scores are better."
        })
    validation_interval: int = attr.ib(default=1000)

    snapshot_interval: int = attr.ib(default=0)

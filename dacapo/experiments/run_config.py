import attr

from .architectures import ArchitectureConfig
from .datasplits import DataSplitConfig
from .tasks import TaskConfig
from .trainers import TrainerConfig


@attr.s
class RunConfig:

    task_config: TaskConfig = attr.ib()
    architecture_config: ArchitectureConfig = attr.ib()
    trainer_config: TrainerConfig = attr.ib()
    datasplit_config: DataSplitConfig = attr.ib()

    name: str = attr.ib(
        metadata={
            "help_text":
                "A unique name for this run. This will be saved so you and "
                "others can find this run. Keep it short and avoid special "
                "characters."
        }
    )

    repetition: int = attr.ib(
        metadata={
            'help_text':
                "The repetition number of this run."
        })
    num_iterations: int = attr.ib(
        metadata={
            'help_text':
                "The number of iterations to train for."
        })

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
    validation_interval: int = attr.ib(
        default=1000,
        metadata={
            'help_text':
                "How often to perform validation."
        })

    snapshot_interval: int = attr.ib(
        default=0,
        metadata={
            'help_text':
                "How often to save a snapshot of a training batch for "
                "manual inspection."
        })

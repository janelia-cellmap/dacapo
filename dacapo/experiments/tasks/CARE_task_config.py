import attr

from .CARE_task import CARETask
from .task_config import TaskConfig


@attr.s
class CARETaskConfig(TaskConfig):
    """This is a Affinities task config used for generating and
    evaluating voxel affinities for instance segmentations.
    """

    task_type = CARETask

    num_channels: int = attr.ib(
        default=1,
        metadata={
            "help_text": "Number of output channels for the image. "
            "Number of ouptut channels should match the number of channels in the ground truth."
        }
    )

import attr

from .Pix2Pix_task import Pix2PixTask
from .task_config import TaskConfig


@attr.s
class Pix2PixTaskConfig(TaskConfig):
    """This is a Pix2Pix task config used for generating and
    evaluating voxel affinities for instance segmentations.
    """

    task_type = Pix2PixTask
    num_channels: int = attr.ib(
        default=2,
        metadata={
            "help_text": "Number of output channels for the image. "
            "Number of ouptut channels should match the number of channels in the ground truth."
        })
        
    dims: int = attr.ib(
        default=2,
        metadata={
            "help_text": "Number of UNet dimensions. "
            "Number of dimensions should match the number of channels in the ground truth."
        }
    )

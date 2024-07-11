import attr

from .array_config import ArrayConfig
from .rescaled_array import RescaledArray


@attr.s
class RescaledArrayConfig(ArrayConfig):
    """
    A configuration for a RescaledArray. This array will up or down sample an array into the desired voxel size.

    Attributes:
        source_array_config (ArrayConfig): The Array that you want to upsample or downsample.
        target_voxel_size (tuple): The target voxel size.
        interp_order (bool): The order of the interpolation.
    Methods:
        create_array: Creates a RescaledArray from the configuration.
    Note:
        This class is meant to be used with the ArrayDataset class.

    """

    array_type = RescaledArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array that you want to upsample or downsample."}
    )

    target_voxel_size: tuple = attr.ib(
        metadata={"help_text": "The target voxel size."}
    )
    interp_order: bool = attr.ib(
        metadata={"help_text": "The order of the interpolation."}
    )

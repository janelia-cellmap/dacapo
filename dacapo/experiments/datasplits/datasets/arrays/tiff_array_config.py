import attr

from .array_config import ArrayConfig

from funlib.geometry import Coordinate

from upath import UPath as Path
from typing import List


@attr.s
class TiffArrayConfig(ArrayConfig):
    """
    This config class provides the necessary configuration for a tiff array

    Attributes:
        file_name: Path
            The file name of the tiff.
        offset: Coordinate
            The offset of the array.
        voxel_size: Coordinate
            The voxel size of the array.
        axis_names: List[str]
            The axis_names of the array.
    Note:
        This class is a subclass of ArrayConfig.
    """

    file_name: Path = attr.ib(
        metadata={"help_text": "The file name of the zarr container."}
    )
    offset: Coordinate = attr.ib(
        metadata={
            "help_text": "The offset for this array. This must be provided "
            "to align this array with any others provided."
        }
    )
    voxel_size: Coordinate = attr.ib(
        metadata={"help_text": "The size of each voxel in each dimension."}
    )
    axis_names: List[str] = attr.ib(metadata={"help_text": "The axis_names of your array"})

import attr

from .array_config import ArrayConfig
from .tiff_array import TiffArray

from funlib.geometry import Coordinate

from upath import UPath as Path
from typing import List


@attr.s
class ZarrArrayConfig(ArrayConfig):
    array_type = TiffArray

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
    axes: List[str] = attr.ib(metadata={"help_text": "The axes of your array"})

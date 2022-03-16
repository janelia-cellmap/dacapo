import attr

from .array_config import ArrayConfig
from .zarr_array import ZarrArray

from funlib.geometry import Coordinate

from pathlib import Path

from typing import Optional, List


@attr.s
class ZarrArrayConfig(ArrayConfig):
    """This config class provides the necessary configuration for a zarr array"""

    array_type = ZarrArray

    file_name: Path = attr.ib(
        metadata={"help_text": "The file name of the zarr container."}
    )
    dataset: str = attr.ib(
        metadata={
            "help_text": "The name of your dataset. May include '/' characters for nested heirarchies"
        }
    )
    snap_to_grid: Optional[Coordinate] = attr.ib(
        default=None,
        metadata={
            "help_text": "If you need to make sure your ROI's align with a specific voxel_size"
        },
    )
    _axes: Optional[List[str]] = attr.ib(
        default=None, metadata={"help_text": "The axes of your data!"}
    )

import attr

from .array_config import ArrayConfig
from .resampled_array import ResampledArray
from .array_config import ArrayConfig

from funlib.geometry import Coordinate


@attr.s
class ResampledArrayConfig(ArrayConfig):
    """This array will up or down sample an array into the desired voxel size."""

    array_type = ResampledArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    upsample: Coordinate = attr.ib()
    downsample: Coordinate = attr.ib()
    interp_order: bool = attr.ib()

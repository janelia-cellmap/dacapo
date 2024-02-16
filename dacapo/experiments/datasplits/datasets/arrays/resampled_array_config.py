import attr

from .array_config import ArrayConfig
from .resampled_array import ResampledArray

from funlib.geometry import Coordinate


@attr.s
class ResampledArrayConfig(ArrayConfig):
    """A class representing the configuration for resampling a source array.

    This class facilitates upsampling or downsampling of a source array 
    to achieve the desired voxel size. The configuration required for 
    resampling includes parameters for the source array, upsampling 
    coordinate, downsampling coordinate, and interpolation order.

    Attributes:
        array_type: A class object representing ResampledArray type.
        source_array_config (ArrayConfig): Configuration of the source array to be resampled.
        upsample (Coordinate): Coordinate for the amount to upsample the array.
        downsample (Coordinate): Coordinate for the amount to downsample the array.
        interp_order (bool): Order of interpolation applied during resampling.

    """
    array_type = ResampledArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array that you want to upsample or downsample."}
    )

    upsample: Coordinate = attr.ib(
        metadata={"help_text": "The amount by which to upsample!"}
    )
    downsample: Coordinate = attr.ib(
        metadata={"help_text": "The amount by which to downsample!"}
    )
    interp_order: bool = attr.ib(
        metadata={"help_text": "The order of the interpolation!"}
    )
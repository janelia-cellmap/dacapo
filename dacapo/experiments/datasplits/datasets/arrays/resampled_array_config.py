import attr

from .array_config import ArrayConfig
from .resampled_array import ResampledArray

from funlib.geometry import Coordinate


@attr.s
class ResampledArrayConfig(ArrayConfig):
    

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

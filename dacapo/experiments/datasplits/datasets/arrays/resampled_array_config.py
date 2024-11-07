import attr

from .array_config import ArrayConfig

from funlib.geometry import Coordinate
from funlib.persistence import Array


@attr.s
class ResampledArrayConfig(ArrayConfig):
    """
    A configuration for a ResampledArray. This array will up or down sample an array into the desired voxel size.

    Attributes:
        source_array_config (ArrayConfig): The Array that you want to upsample or downsample.
        upsample (Coordinate): The amount by which to upsample!
        downsample (Coordinate): The amount by which to downsample!
        interp_order (bool): The order of the interpolation!
    Methods:
        create_array: Creates a ResampledArray from the configuration.
    Note:
        This class is meant to be used with the ArrayDataset class.

    """

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

    def array(self, mode: str = "r") -> Array:
        # This is non trivial. We want to upsample or downsample the source
        # array lazily. Not entirely sure how to do this with dask arrays.
        raise NotImplementedError()

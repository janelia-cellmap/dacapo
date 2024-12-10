import attr

from .array_config import ArrayConfig

from funlib.geometry import Coordinate
from funlib.persistence import Array

from xarray_multiscale.multiscale import downscale_dask
from xarray_multiscale import windowed_mean
import numpy as np
import dask.array as da
from skimage.transform import rescale

from typing import Sequence


def adjust_shape(array: da.Array, scale_factors: Sequence[int]) -> da.Array:
    """
    Crop array to a shape that is a multiple of the scale factors.
    This allows for clean downsampling.
    """
    misalignment = np.any(np.mod(array.shape, scale_factors))
    if misalignment:
        new_shape = np.subtract(array.shape, np.mod(array.shape, scale_factors))
        slices = tuple(slice(0, s) for s in new_shape)
        array = array[slices]
    return array


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

    def preprocess(self, array: Array) -> Array:
        """
        Preprocess an array by resampling it to the desired voxel size.
        """
        if self.downsample is not None:
            downsample = list(self.downsample)
            for i, axis_name in enumerate(array.axis_names):
                if "^" in axis_name:
                    downsample = downsample[:i] + [1] + downsample[i:]
            return Array(
                data=downscale_dask(
                    adjust_shape(array.data, downsample),
                    windowed_mean,
                    scale_factors=downsample,
                ),
                offset=array.offset,
                voxel_size=array.voxel_size * self.downsample,
                axis_names=array.axis_names,
                units=array.units,
            )
        elif self.upsample is not None:
            upsample = list(self.upsample)
            for i, axis_name in enumerate(array.axis_names):
                if "^" in axis_name:
                    upsample = upsample[:i] + [1] + upsample[i:]

            depth = [int(x > 1) for x in upsample]
            trim_slicing = tuple(
                slice(d * s, (-d * s)) if d > 1 else slice(None)
                for d, s in zip(depth, upsample)
            )

            rescaled_arr = da.map_overlap(
                lambda x: rescale(
                    x, upsample, order=int(self.interp_order), preserve_range=True
                )[trim_slicing],
                array.data,
                depth=depth,
                boundary="reflect",
                trim=False,
                dtype=array.data.dtype,
                chunks=tuple(c * u for c, u in zip(array.data.chunksize, upsample)),
            )

            return Array(
                data=rescaled_arr,
                offset=array.offset,
                voxel_size=array.voxel_size / self.upsample,
                axis_names=array.axis_names,
                units=array.units,
            )

    def array(self, mode: str = "r") -> Array:
        source_array = self.source_array_config.array(mode)

        return self.preprocess(source_array)

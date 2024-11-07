import attr

from .array_config import ArrayConfig
from funlib.persistence import Array
import dask.array as da
from dacapo.tmp import num_channels_from_array


@attr.s
class LogicalOrArrayConfig(ArrayConfig):
    """
    This config class takes a source array and performs a logical or over the channels.
    Good for union multiple masks.

    Attributes:
        source_array_config (ArrayConfig): The Array of masks from which to take the union
    Methods:
        to_array: Returns the LogicalOrArray object
    Notes:
        The source_array_config must be an ArrayConfig object.
    """

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array of masks from which to take the union"}
    )

    def array(self, mode: str = "r") -> Array:
        array = self.source_array_config.array(mode)

        assert num_channels_from_array(array) is not None

        out_array = Array(
            da.zeros(*array.physical_shape, dtype=array.dtype),
            offset=array.offset,
            voxel_size=array.voxel_size,
            axis_names=array.axis_names[1:],
            units=array.units,
        )

        out_array.data = da.maximum(array.data, axis=0)

        # mark data as non-writable
        out_array.lazy_op(lambda data: data)
        return out_array

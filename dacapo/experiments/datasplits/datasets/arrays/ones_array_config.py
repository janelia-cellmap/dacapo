import attr

from .array_config import ArrayConfig
import dask.array as da
from funlib.persistence import Array


@attr.s
class OnesArrayConfig(ArrayConfig):
    """
    This array read data from the source array and then return a np.ones_like() version.

    This is useful for creating a mask array from a source array. For example, if you have a
    2D array of data and you want to create a mask array that is the same shape as the data
    array, you can use this class to create the mask array.

    Attributes:
        source_array_config: The source array that you want to copy and fill with ones.
    Methods:
        create_array: Create the array.
    Note:
        This class is a subclass of ArrayConfig.
    """

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array that you want to copy and fill with ones."}
    )

    def array(self, mode: str = "r") -> Array:
        source_array = self.source_array_config.array(mode)
        return Array(
            data=da.ones(source_array.shape, dtype=source_array.dtype),
            offset=source_array.offset,
            voxel_size=source_array.voxel_size,
            axis_names=source_array.axis_names,
            units=source_array.units,
        )

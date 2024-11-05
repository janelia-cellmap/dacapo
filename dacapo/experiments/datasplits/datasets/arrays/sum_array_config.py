import attr

from .array_config import ArrayConfig

from typing import List
from funlib.persistence import Array
import dask.array as da


@attr.s
class SumArrayConfig(ArrayConfig):
    """
    This config class provides the necessary configuration for a sum
    array.

    Attributes:
        source_array_configs: List[ArrayConfig]
            The Array of masks from which to take the union
    Note:
        This class is a subclass of ArrayConfig.
    """

    source_array_configs: List[ArrayConfig] = attr.ib(
        metadata={"help_text": "The Array of masks from which to take the union"}
    )

    def array(self, mode: str = "r") -> Array:
        arrays = [
            source_array.array(mode) for source_array in self.source_array_configs
        ]
        return Array(
            data=da.stack([array.data for array in arrays], axis=0).sum(axis=0),
            offset=arrays[0].offset,
            voxel_size=arrays[0].voxel_size,
            axis_names=arrays[0].axis_names,
            units=arrays[0].units,
        )

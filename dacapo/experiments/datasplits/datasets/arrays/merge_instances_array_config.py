import attr

from .array_config import ArrayConfig
from funlib.persistence import Array
from typing import List

import dask.array as da


@attr.s
class MergeInstancesArrayConfig(ArrayConfig):
    """
    Configuration for an array that merges instances from multiple arrays
    into a single array. The instances are merged by taking the union of the
    instances in the source arrays.

    Attributes:
        source_array_configs: List[ArrayConfig]
            The Array of masks from which to take the union
    Methods:
        create_array: () -> MergeInstancesArray
            Create a MergeInstancesArray instance from the configuration
    Notes:
        The MergeInstancesArrayConfig class is used to create a MergeInstancesArray
    """

    source_array_configs: List[ArrayConfig] = attr.ib(
        metadata={"help_text": "The Array of masks from which to take the union"}
    )

    def array(self, mode: str = "r") -> Array:
        arrays = [
            source_array.array(mode) for source_array in self.source_array_configs
        ]
        merged_data = da.stack([array.data for array in arrays], axis=0).sum(axis=0)
        return Array(
            data=merged_data,
            offset=arrays[0].offset,
            voxel_size=arrays[0].voxel_size,
            axis_names=arrays[0].axis_names,
            units=arrays[0].units,
        )

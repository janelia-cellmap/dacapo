import attr

from .array_config import ArrayConfig
from funlib.persistence import Array

from typing import List, Tuple
from dacapo.tmp import num_channels_from_array

import dask.array as da
import numpy as np


@attr.s
class BinarizeArrayConfig(ArrayConfig):
    """
    This config class provides the necessary configuration for turning an Annotated dataset into a
    multi class binary classification problem. Each class will be binarized into a separate channel.

    Attributes:
        source_array_config (ArrayConfig): The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension
        groupings (List[Tuple[str, List[int]]]): List of id groups with a symantic name. Each id group is a List of ids.
            Group i found in groupings[i] will be binarized and placed in channel i.
            An empty group will binarize all non background labels.
        background (int): The id considered background. Will never be binarized to 1, defaults to 0.
    Note:
        This class is used to create a BinarizeArray object which is used to turn an Annotated dataset into a multi class binary classification problem.
        Each class will be binarized into a separate channel.

    """

    source_array_config: ArrayConfig = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    groupings: List[Tuple[str, List[int]]] = attr.ib(
        metadata={
            "help_text": "List of id groups with a symantic name. Each id group is a List of ids. "
            "Group i found in groupings[i] will be binarized and placed in channel i. "
            "An empty group will binarize all non background labels."
        }
    )

    background: int = attr.ib(
        default=0,
        metadata={
            "help_text": "The id considered background. Will never be binarized to 1, defaults to 0."
        },
    )

    def array(self, mode="r") -> Array:
        array = self.source_array_config.array(mode)
        num_channels = num_channels_from_array(array)
        assert num_channels is None, "Input labels cannot have a channel dimension"

        def group_array(data):
            out = da.zeros((len(self.groupings), *array.physical_shape), dtype=np.uint8)
            for i, (_, group_ids) in enumerate(self.groupings):
                if len(group_ids) == 0:
                    out[i] = data != self.background
                else:
                    out[i] = da.isin(data, group_ids)
            return out

        data = group_array(array.data)
        out_array = Array(
            data,
            array.offset,
            array.voxel_size,
            ["c^"] + list(array.axis_names),
            units=array.units,
        )

        # callable lazy op so funlib.persistence doesn't try to recoginize this data as writable
        out_array.lazy_op(lambda data: data)

        return out_array

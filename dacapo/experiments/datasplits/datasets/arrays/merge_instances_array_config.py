import attr

from .array_config import ArrayConfig
from funlib.persistence import Array
from typing import List


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
        raise NotImplementedError
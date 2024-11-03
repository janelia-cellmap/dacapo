import attr

from .array_config import ArrayConfig
from .merge_instances_array import MergeInstancesArray

from typing import List


@attr.s
class MergeInstancesArrayConfig(ArrayConfig):
    

    array_type = MergeInstancesArray

    source_array_configs: List[ArrayConfig] = attr.ib(
        metadata={"help_text": "The Array of masks from which to take the union"}
    )

import attr

from .array_config import ArrayConfig
from .sum_array import SumArray

from typing import List


@attr.s
class SumArrayConfig(ArrayConfig):
    

    array_type = SumArray

    source_array_configs: List[ArrayConfig] = attr.ib(
        metadata={"help_text": "The Array of masks from which to take the union"}
    )

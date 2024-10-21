import attr

from .array_config import ArrayConfig

from typing import List


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

import attr

from .array_config import ArrayConfig
from .intensity_array import IntensitiesArray
from .array_config import ArrayConfig

from typing import List, Tuple


@attr.s
class IntensitiesArrayConfig(ArrayConfig):
    """This config class provides the necessary configuration for turning an Annotated dataset into a
    multi class binary classification problem"""

    array_type = IntensitiesArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    min: float = attr.ib(
        metadata={
            "help_text": "The minimum intensity in your data"
        }
    )
    max: float = attr.ib(
        metadata={
            "help_text": "The maximum intensity in your data"
        }
    )

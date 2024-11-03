import attr

from .array_config import ArrayConfig
from .intensity_array import IntensitiesArray


@attr.s
class IntensitiesArrayConfig(ArrayConfig):
    

    array_type = IntensitiesArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    min: float = attr.ib(metadata={"help_text": "The minimum intensity in your data"})
    max: float = attr.ib(metadata={"help_text": "The maximum intensity in your data"})

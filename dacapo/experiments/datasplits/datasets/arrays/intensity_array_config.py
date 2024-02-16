import attr

from .array_config import ArrayConfig
from .intensity_array import IntensitiesArray


@attr.s
class IntensitiesArrayConfig(ArrayConfig):
    """Generates configurations for the creation of Intensity array.

    This class is a child class of ArrayConfig that holds attributes for IntensitiesArray.
    Also inherits the methods of ArrayConfig to utilize for IntensitiesArray.

    Attributes:
        array_type: The class IntensitiesArray.
        source_array_config: Object of ArrayConfig that holds the generic settings for an array.
        min: Float. The minimum intensity in the data.
        max: Float. The maximum intensity in the data.
    """
    
    array_type = IntensitiesArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    min: float = attr.ib(metadata={"help_text": "The minimum intensity in your data"})
    max: float = attr.ib(metadata={"help_text": "The maximum intensity in your data"})
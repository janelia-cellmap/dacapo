import attr

from .array_config import ArrayConfig
from .intensity_array import IntensitiesArray


@attr.s
class IntensitiesArrayConfig(ArrayConfig):
    """
    This config class provides the necessary configuration for turning an Annotated dataset into a
    multi class binary classification problem. It takes a source array and normalizes the intensities
    between 0 and 1. The source array is expected to contain a volume with uint64 voxels and no channel dimension.
        
    Attributes:
        source_array_config (ArrayConfig): The Array from which to pull annotated data
        min (float): The minimum intensity in your data
        max (float): The maximum intensity in your data
    Methods:
        to_array: Returns the IntensitiesArray object
    Notes:
        The source_array_config must be an ArrayConfig object.
    """

    array_type = IntensitiesArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    min: float = attr.ib(metadata={"help_text": "The minimum intensity in your data"})
    max: float = attr.ib(metadata={"help_text": "The maximum intensity in your data"})

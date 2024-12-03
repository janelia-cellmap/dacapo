import attr
from funlib.persistence import Array

from .array_config import ArrayConfig


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

    source_array_config: ArrayConfig = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    min: float = attr.ib(metadata={"help_text": "The minimum intensity in your data"})
    max: float = attr.ib(metadata={"help_text": "The maximum intensity in your data"})

    def array(self, mode: str = "r") -> Array:
        array = self.source_array_config.array(mode)
        array.lazy_op(lambda data: (data - self.min) / (self.max - self.min))
        return array

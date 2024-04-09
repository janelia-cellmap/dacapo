import attr

from .array_config import ArrayConfig
from .logical_or_array import LogicalOrArray


@attr.s
class LogicalOrArrayConfig(ArrayConfig):
    """
    This config class takes a source array and performs a logical or over the channels.
    Good for union multiple masks.

    Attributes:
        source_array_config (ArrayConfig): The Array of masks from which to take the union
    Methods:
        to_array: Returns the LogicalOrArray object
    Notes:
        The source_array_config must be an ArrayConfig object.
    """

    array_type = LogicalOrArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array of masks from which to take the union"}
    )

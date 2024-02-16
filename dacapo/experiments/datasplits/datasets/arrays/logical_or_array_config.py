import attr

from .array_config import ArrayConfig
from .logical_or_array import LogicalOrArray

@attr.s
class LogicalOrArrayConfig(ArrayConfig):
    """
    A Config class inherited from ArrayConfig. This is specifically used for creating a boolean 
    array with 'logical or' comparisons across the array's elements.

    Attributes:
        array_type (obj): LogicalOrArray object is passed as the array_type argument.
        source_array_config (ArrayConfig): The array configuration from which union of masks will be created. 

    Metadata:
        help_text: A short description of the source_array_config attribute.
    """
    
    array_type = LogicalOrArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array of masks from which to take the union"}
    )
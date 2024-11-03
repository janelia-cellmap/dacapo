import attr

from .array_config import ArrayConfig
from .logical_or_array import LogicalOrArray


@attr.s
class LogicalOrArrayConfig(ArrayConfig):
    array_type = LogicalOrArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array of masks from which to take the union"}
    )

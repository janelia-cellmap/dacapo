import attr

from .array_config import ArrayConfig
from .constant_array import ConstantArray


@attr.s
class ConstantArrayConfig(ArrayConfig):
    array_type = ConstantArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array that you want to copy and fill with ones."}
    )

    constant: int = attr.ib(
        metadata={"help_text": "The constant value to fill the array with."}, default=1
    )

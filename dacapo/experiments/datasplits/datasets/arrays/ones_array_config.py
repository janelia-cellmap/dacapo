import attr

from .array_config import ArrayConfig
from .ones_array import OnesArray


@attr.s
class OnesArrayConfig(ArrayConfig):
    """This array read data from the source array and then return a np.ones_like() version."""

    array_type = OnesArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array that you want to copy and fill with ones."}
    )

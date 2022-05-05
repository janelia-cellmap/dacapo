import attr

from .array_config import ArrayConfig
from .concat_array import ConcatArray
from .array_config import ArrayConfig

from typing import List, Dict, Optional


@attr.s
class ConcatArrayConfig(ArrayConfig):
    """This array read data from the source array and then return a np.ones_like() version."""

    array_type = ConcatArray

    channels: List[str] = attr.ib(
        metadata={"help_text": "An ordering for the source_arrays."}
    )
    source_array_configs: Dict[str, ArrayConfig] = attr.ib(
        metadata={
            "help_text": "A mapping from channels to array_configs. If a channel "
            "has no ArrayConfig it will be filled with zeros"
        }
    )
    default_config: Optional[ArrayConfig] = attr.ib(
        default=None,
        metadata={
            "help_text": "An optional array providing the default array per channel. If "
            "not provided, missing channels will simply be filled with 0s"
        },
    )

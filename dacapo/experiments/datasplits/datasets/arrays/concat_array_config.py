import attr

from .array_config import ArrayConfig
from .concat_array import ConcatArray

from typing import List, Dict, Optional


@attr.s
class ConcatArrayConfig(ArrayConfig):
    """
    This array read data from the source array and then return a np.ones_like() version of the data.

    Attributes:
        channels (List[str]): An ordering for the source_arrays.
        source_array_configs (Dict[str, ArrayConfig]): A mapping from channels to array_configs. If a channel has no ArrayConfig it will be filled with zeros
        default_config (Optional[ArrayConfig]): An optional array providing the default array per channel. If not provided, missing channels will simply be filled with 0s
    Methods:
        __attrs_post_init__(self): This method is called after the instance has been initialized by the constructor. It is used to set the default_config to an instance of ArrayConfig if it is None.
        get_array(self, source_arrays: Dict[str, np.ndarray]) -> np.ndarray: This method reads data from the source array and then return a np.ones_like() version of the data.
    Note:
        This class is used to create a ConcatArray object which is used to read data from the source array and then return a np.ones_like() version of the data.
        The source array is a dictionary with the key being the channel and the value being the array.
    """

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

import attr

from .array_config import ArrayConfig
from .ones_array import OnesArray


@attr.s
class OnesArrayConfig(ArrayConfig):
    """ 
    Creates a OnesArrayConfig object which is a configuration to create a ones array.
    
    Attributes:
        array_type (class): Class type of the array.
        source_array_config (ArrayConfig): Configuration of the source array from which data is read and copied to
                                           create a np.ones_like() version.
    """

    array_type = OnesArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array that you want to copy and fill with ones."}
    )
import attr

from .array_config import ArrayConfig
from funlib.persistence import Array


@attr.s
class ConstantArrayConfig(ArrayConfig):
    """
    This array read data from the source array and then return a np.ones_like() version.

    This is useful for creating a mask array from a source array. For example, if you have a
    2D array of data and you want to create a mask array that is the same shape as the data
    array, you can use this class to create the mask array.

    Attributes:
        source_array_config: The source array that you want to copy and fill with ones.
    Methods:
        create_array: Create the array.
    Note:
        This class is a subclass of ArrayConfig.
    """

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array that you want to copy and fill with ones."}
    )

    constant: int = attr.ib(
        metadata={"help_text": "The constant value to fill the array with."}, default=1
    )

    def array(self, mode: str = "r") -> Array:
        array = self.source_array_config.array(mode)

        def set_constant(array):
            array[:] = self.constant
            return array

        array.lazy_op(set_constant)
        return array

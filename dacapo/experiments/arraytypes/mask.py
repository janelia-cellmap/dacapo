from .arraytype import ArrayType

import attr


@attr.s
class Mask(ArrayType):
    """
    A class that inherits the ArrayType class. This is a representation of a Mask in the system.

    Methods:
        interpolatable():
            It is a method that returns False.
    Note:
        This class is used to represent a Mask object in the system.
    """

    @property
    def interpolatable(self) -> bool:
        """
        Method to return False.

        Returns:
            bool
                Returns a boolean value of False representing that the values are not interpolatable.
        Raises:
            NotImplementedError
                This method is not implemented in this class.
        Examples:
            >>> mask = Mask()
            >>> mask.interpolatable
            False
        Note:
            This method is used to check if the array is interpolatable.
        """
        return False

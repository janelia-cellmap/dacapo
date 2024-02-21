"""
This is a module of the dacapo python library of funkelab that contains the definition of Mask class which inherits the ArrayType class. 

Attributes:
-----------
attr: module
    This is a python library for creating classes and managing attributes and validators. 

Classes:
--------
Mask
    Inherits ArrayType class. This class defines a method called interpolatable, which returns False.
"""

from .arraytype import ArrayType

import attr


@attr.s
class Mask(ArrayType):
    """
    A class that inherits the ArrayType class. This is a representation of a Mask in the system.

    Methods
    -------
    interpolatable():
        It is a method that returns False.
    """

    @property
    def interpolatable(self) -> bool:
        """
        Method to return False.

        Returns
        ------
        bool
            Returns a boolean value of False representing that the values are not interpolatable.
        """
        return False

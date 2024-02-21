"""
This module contains the IntensitiesArray class.

Imported libraries and modules:

    * attr: used for creating classes without having to write explicit `__init__`, `__repr__`, etc. methods.
    * typing: for providing hint types for python objects/functions.

Classes:
    * IntensitiesArray(ArrayType)
"""

from .arraytype import ArrayType
import attr
from typing import Dict

@attr.s
class IntensitiesArray(ArrayType):
    """
    An IntensitiesArray is an Array of measured intensities. Inherits from ArrayType.

    Attributes:
        channels (Dict[int, str]): A mapping from channel to a name describing that channel.
        min (float): The minimum possible value of your intensities.
        max (float): The maximum possible value of your intensities.
        
    The `@property` defined enables to treat the 'interpolatable' as an attribute of the class.
    """

    channels: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to a name describing that channel."
        }
    )
    min: float = attr.ib(
        metadata={"help_text": "The minimum possible value of your intensities."}
    )
    max: float = attr.ib(
        metadata={"help_text": "The maximum possible value of your intensities."}
    )

    @property
    def interpolatable(self) -> bool:
        """
        The metadata information for interpolation ability.
        
        Returns:
            bool: Always returns True for this IntensitiesArray class. The actual functionality depends on the specific implementation.
        """
        return True
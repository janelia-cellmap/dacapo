from .arraytype import ArrayType


import attr

from typing import Dict


@attr.s
class IntensitiesArray(ArrayType):
    """
    An IntensitiesArray is an Array of measured intensities.
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
        return True

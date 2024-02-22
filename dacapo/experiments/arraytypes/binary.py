from .arraytype import ArrayType

import attr

from typing import Dict


@attr.s
class BinaryArray(ArrayType):
    """
    An BinaryArray is a bool or uint8 Array where each
    voxel is either 1 or 0.
    """

    channels: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to class for the binary classification."
        }
    )

    @property
    def interpolatable(self) -> bool:
        return False

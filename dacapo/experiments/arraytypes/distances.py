from arraytype import ArrayType

import attr

from typing import Dict


@attr.s
class DistanceArray(ArrayType):
    """
    An array containing signed distances to the nearest boundary voxel for a particular label class.
    Distances should be positive outside an object and negative inside an object.
    """

    classes: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to class on which distances were calculated"
        }
    )

    @property
    def interpolatable(self) -> bool:
        return True

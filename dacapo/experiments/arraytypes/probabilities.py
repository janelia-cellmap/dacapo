from .arraytype import ArrayType

import attr

from typing import List


@attr.s
class ProbabilityArray(ArrayType):
    """
    An array containing probabilities for each voxel. I.e. each voxel has a vector
    of length `c` where `c` is the number of classes. The l1 norm of this vector should
    always be 1. The class of this voxel can be determined by simply taking the
    argmax.
    """

    classes: List[str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to class on which distances were calculated"
        }
    )

    @property
    def interpolatable(self) -> bool:
        return True

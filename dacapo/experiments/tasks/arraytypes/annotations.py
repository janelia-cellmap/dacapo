from .arraytype import ArrayType

import attr
from typing import Dict


@attr.s
class AnnotationArray(ArrayType):
    """
    An AnnotationArray is a uint8, uint16, uint32 or uint64 Array where each
    voxel has a value associated with its class.
    """

    classes: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from class label to class name. "
            "For example {1:'mitochondria', 2:'membrane'} etc."
        }
    )

    @property
    def interpolatable(self):
        return False

from .arraytype import ArrayType

import attr
from typing import Dict


@attr.s
class AnnotationArray(ArrayType):
    classes: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from class label to class name. "
            "For example {1:'mitochondria', 2:'membrane'} etc."
        }
    )

    @property
    def interpolatable(self):
        return False

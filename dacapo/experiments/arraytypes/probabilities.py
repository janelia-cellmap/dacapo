from .arraytype import ArrayType
import attr
from typing import List


@attr.s
class ProbabilityArray(ArrayType):
    

    classes: List[str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to class on which distances were calculated"
        }
    )

    @property
    def interpolatable(self) -> bool:
        
        return True

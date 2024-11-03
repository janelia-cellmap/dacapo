from .arraytype import ArrayType

import attr

from typing import Dict


@attr.s
class DistanceArray(ArrayType):
    

    classes: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to class on which distances were calculated"
        }
    )

    @property
    def interpolatable(self) -> bool:
        
        return True

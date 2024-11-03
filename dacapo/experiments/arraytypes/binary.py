from .arraytype import ArrayType

import attr

from typing import Dict


@attr.s
class BinaryArray(ArrayType):
    channels: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to class for the binary classification."
        }
    )

    @property
    def interpolatable(self) -> bool:
        return False

import attr

from typing import Tuple


@attr.s
class ArrayConfig:
    

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this array. This will be saved so you "
            "and others can find and reuse this array. Keep it short "
            "and avoid special characters."
        }
    )

    def verify(self) -> Tuple[bool, str]:
        
        return True, "No validation for this Array"

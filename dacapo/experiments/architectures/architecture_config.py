import attr
from typing import Tuple


@attr.s
class ArchitectureConfig:
    

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this architecture. This will be saved so "
            "you and others can find and reuse this task. Keep it short "
            "and avoid special characters."
        }
    )

    def verify(self) -> Tuple[bool, str]:
        
        return True, "No validation for this Architecture"

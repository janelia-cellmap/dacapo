from .arraytype import ArrayType

import attr


@attr.s
class Mask(ArrayType):
    @property
    def interpolatable(self) -> bool:
        return False

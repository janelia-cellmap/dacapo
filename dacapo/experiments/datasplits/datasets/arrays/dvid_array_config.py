import attr

from .array_config import ArrayConfig
from .dvid_array import DVIDArray


from typing import Tuple


@attr.s
class DVIDArrayConfig(ArrayConfig):
    array_type = DVIDArray

    source: Tuple[str, str, str] = attr.ib(
        metadata={"help_text": "The source strings."}
    )

    def verify(self) -> Tuple[bool, str]:
        return True, "No validation for this Array"

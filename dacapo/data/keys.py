from enum import Enum
from typing import Union


class ArrayKey(Enum):
    RAW = "raw"
    GT = "gt"
    MASK = "mask"
    NON_EMPTY = "NON_EMPTY"


class GraphKey(Enum):
    NON_EMPTY = "non_empty"


DataKey = Union[ArrayKey, GraphKey]
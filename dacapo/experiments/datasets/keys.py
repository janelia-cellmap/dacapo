from enum import Enum
from typing import Union


class ArrayKey(Enum):

    RAW = "raw"
    GT = "gt"
    MASK = "mask"
    NON_EMPTY = "non_empty_mask"


class GraphKey(Enum):

    SPECIFIED_LOCATIONS = "specified_locations"


DataKey = Union[ArrayKey, GraphKey]

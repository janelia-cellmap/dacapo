from dacapo.converter import converter

from enum import Enum
from typing import Union, get_args


class ArrayKey(Enum):
    RAW = "raw"
    GT = "gt"
    MASK = "mask"
    NON_EMPTY = "non_empty_mask"


class GraphKey(Enum):
    SPECIFIED_LOCATIONS = "specified_locations"


DataKey = Union[ArrayKey, GraphKey]


converter.register_unstructure_hook(
    DataKey,
    lambda o: {"__type__": type(o).__name__, "value": converter.unstructure(o)},
)
converter.register_structure_hook(
    DataKey,
    lambda o, _: eval(o.pop("__type__"))(o["value"]),
)

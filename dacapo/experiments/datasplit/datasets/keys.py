from enum import Enum, unique


class DataKey(Enum):
    pass


@unique
class ArrayKey(DataKey):

    RAW = "raw"
    GT = "gt"
    MASK = "mask"
    NON_EMPTY = "non_empty_mask"


@unique
class GraphKey(DataKey):

    SPECIFIED_LOCATIONS = "specified_locations"

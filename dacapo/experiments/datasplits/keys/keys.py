from enum import Enum, unique


class DataKey(Enum):
    """
    Represent a base class for various types of keys in Dacapo library.

    Attributes:
        RAW: str
            The raw data key.
        GT: str
            The ground truth data key.
        MASK: str
            The data mask key.
        NON_EMPTY: str
            The data key for non-empty mask.
        SPECIFIED_LOCATIONS: str
            The key for specified locations in the graph.
    Methods:
        __str__():
            Return the string representation of the key.
    Notes:
        This class is a base class and should not be instantiated.
    """

    pass


@unique
class ArrayKey(DataKey):
    """
    A unique enumeration representing different types of array keys

    Attributes:
        RAW: str
            The raw data key.
        GT: str
            The ground truth data key.
        MASK: str
            The data mask key.
        NON_EMPTY: str
            The data key for non-empty mask.
    Methods:
        __str__():
            Return the string representation of the key.
    Notes:
        This class is a base class and should not be instantiated.
    """

    RAW = "raw"
    GT = "gt"
    MASK = "mask"
    NON_EMPTY = "non_empty_mask"


@unique
class GraphKey(DataKey):
    """
    A unique enumeration representing different types of graph keys

    Attributes:
        SPECIFIED_LOCATIONS: str
            The key for specified locations in the graph.
    Methods:
        __str__():
            Return the string representation of the key.
    Notes:
        This class is a base class and should not be instantiated.
    """

    SPECIFIED_LOCATIONS = "specified_locations"

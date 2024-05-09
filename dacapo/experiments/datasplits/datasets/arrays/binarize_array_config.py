import attr

from .array_config import ArrayConfig
from .binarize_array import BinarizeArray

from typing import List, Tuple


@attr.s
class BinarizeArrayConfig(ArrayConfig):
    """
    This config class provides the necessary configuration for turning an Annotated dataset into a
    multi class binary classification problem. Each class will be binarized into a separate channel.

    Attributes:
        source_array_config (ArrayConfig): The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension
        groupings (List[Tuple[str, List[int]]]): List of id groups with a symantic name. Each id group is a List of ids.
            Group i found in groupings[i] will be binarized and placed in channel i.
            An empty group will binarize all non background labels.
        background (int): The id considered background. Will never be binarized to 1, defaults to 0.
    Note:
        This class is used to create a BinarizeArray object which is used to turn an Annotated dataset into a multi class binary classification problem.
        Each class will be binarized into a separate channel.

    """

    array_type = BinarizeArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    groupings: List[Tuple[str, List[int]]] = attr.ib(
        metadata={
            "help_text": "List of id groups with a symantic name. Each id group is a List of ids. "
            "Group i found in groupings[i] will be binarized and placed in channel i. "
            "An empty group will binarize all non background labels."
        }
    )

    background: int = attr.ib(
        default=0,
        metadata={
            "help_text": "The id considered background. Will never be binarized to 1, defaults to 0."
        },
    )

import attr

from .array_config import ArrayConfig
from .binarize_array import BinarizeArray

from typing import List, Tuple


@attr.s
class BinarizeArrayConfig(ArrayConfig):
    """
    The BinarizeArrayConfig class provides configuration settings to transform
    an annotated dataset into a binary classification problem for multiple classes.

    This config class uses a BinaryArray type to store the array values and applies
    transformations based on groups of IDs.

    Attributes:
        array_type (class): The array type to use for the logic. It is a BinaryArray.
        source_array_config (ArrayConfig): The configuration from which to get annotated data.
            This configuration is expected to contain a volume with uint64 voxels with no channel dimension.
        groupings (List[Tuple[str, List[int]]]): List of groups of IDs, each with a semantic name.
            Each ID group is a list of IDs. The IDs in group 'i' in 'groupings[i]' will be binarized
            and placed in channel 'i'. An empty group will contain all non-background labels binarized.
        background (int, optional): The ID considered to be the 'background'. This ID will never be binarized to 1.
            Defaults to 0.
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
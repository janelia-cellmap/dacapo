import attr

from .array_config import ArrayConfig
from .missing_annotations_mask import MissingAnnotationsMask

from typing import List, Tuple


@attr.s
class MissingAnnotationsMaskConfig(ArrayConfig):
    """This config class provides the necessary configuration for turning an Annotated dataset into a
    multi class binary classification problem"""

    array_type = MissingAnnotationsMask

    source_array_config: ArrayConfig = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    groupings: List[Tuple[str, List[int]]] = attr.ib(
        metadata={
            "help_text": "List of id groups with a symantic name. Each id group is a List of ids. "
            "Group i found in groupings[i] will be binarized and placed in channel i."
        }
    )

import attr

from .array_config import ArrayConfig
from .missing_annotations_mask import MissingAnnotationsMask

from typing import List, Tuple


@attr.s
class MissingAnnotationsMaskConfig(ArrayConfig):
    """A configuration class for handling missing annotations in an array.

    This class extends the ArrayConfig class for specialized handling of arrays from 
    annotated datasets. It aids in transforming Annotated dataset into a multi-class 
    binary classification problem.

    Attributes:
        array_type: Type of the array which is MissingAnnotationsMask for this class.
        source_array_config: The ArrayConfig object from which to pull annotated data.
        groupings: List of groupings where each group has a semantic name and a list of ids.
                   Each group is binarized and placed in its respective channel. 

    Metadata:
        source_array_config: Expect an array with uint64 voxels and no channel dimension.
        groupings: Groups with ids are defined here. The ith group will be binarized and 
                   placed in the ith channel.
    """

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
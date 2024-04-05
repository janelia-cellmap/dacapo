from .arraytype import ArrayType

import attr
from typing import Dict


@attr.s
class AnnotationArray(ArrayType):
    """
    An AnnotationArray is a uint8, uint16, uint32 or uint64 Array where each
    voxel has a value associated with its class. The class of each voxel can be
    determined by simply taking the value.

    Attributes:
        classes (Dict[int, str]): A mapping from class label to class name.
    Methods:
        interpolatable(self) -> bool: It is a method that returns False.
    Note:
        This class is used to create an AnnotationArray object which is used to represent an array of class labels.
    """

    classes: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from class label to class name. "
            "For example {1:'mitochondria', 2:'membrane'} etc."
        }
    )

    @property
    def interpolatable(self):
        """
        Method to return False.

        Returns:
            bool
                Returns a boolean value of False representing that the values are not interpolatable.
        Raises:
            NotImplementedError
                This method is not implemented in this class.
        Examples:
            >>> annotation_array = AnnotationArray(classes={1: "mitochondria", 2: "membrane"})
            >>> annotation_array.interpolatable
            False
        Note:
            This method is used to check if the array is interpolatable.
        """
        return False

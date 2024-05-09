from .arraytype import ArrayType
import attr
from typing import List


@attr.s
class ProbabilityArray(ArrayType):
    """
    Class to represent an array containing probability distributions for each voxel pointed by its coordinate.

    The class defines a ProbabilityArray object with each voxel having a vector of length `c`, where `c` is the
    number of classes. The l1 norm of this vector should always be 1. The class of each voxel can be
    determined by simply taking the argmax.

    Attributes:
        classes (List[str]): A mapping from channel to class on which distances were calculated.
    Note:
        This class is used to create a ProbabilityArray object which is used to represent an array containing probability distributions for each voxel pointed by its coordinate.
        The class of each voxel can be determined by simply taking the argmax.
    """

    classes: List[str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to class on which distances were calculated"
        }
    )

    @property
    def interpolatable(self) -> bool:
        """
        Checks if the array is interpolatable. Returns True for this class.

        Returns:
            bool: True indicating that the data can be interpolated.
        Raises:
            NotImplementedError: This method is not implemented in this class
        Examples:
            >>> probability_array = ProbabilityArray(classes=["class1", "class2"])
            >>> probability_array.interpolatable
            True
        Note:
            This method is used to check if the array is interpolatable.
        """
        return True

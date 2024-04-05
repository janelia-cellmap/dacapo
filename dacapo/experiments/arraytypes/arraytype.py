from abc import ABC, abstractmethod


# TODO: Should be read only
class ArrayType(ABC):
    """
    The type of data provided by an array. The ArrayType class helps to keep
    track of the semantic meaning of an Array. Additionally the ArrayType
    keeps track of metadata that is specific to this datatype such as
    num_classes for an annotated volume or channel names for intensity
    arrays. The ArrayType class is an abstract class and should be subclassed
    to represent different types of arrays.

    Attributes:
        num_classes (int): The number of classes in the array.
        channel_names (List[str]): The names of the channels in the array.
    Methods:
        interpolatable: This is an abstract method which should be overridden in each of the subclasses to determine if an array is interpolatable or not.
    Note:
        This class is used to create an ArrayType object which is used to represent the type of data provided by an array.
    """

    @property
    @abstractmethod
    def interpolatable(self) -> bool:
        """
        This is an abstract method which should be overridden in each of the subclasses
        to determine if an array is interpolatable or not.

        Returns:
            bool: True if the array is interpolatable, False otherwise.
        Raises:
            NotImplementedError: This method is not implemented in this class.
        Examples:
            >>> array_type = ArrayType()
            >>> array_type.interpolatable
            NotImplementedError
        Note:
            This method is used to check if the array is interpolatable.
        """
        pass

from abc import ABC, abstractmethod


# TODO: Should be read only
class ArrayType(ABC):
    """
    The type of data provided by an array. The ArrayType class helps to keep
    track of the semantic meaning of an Array. Additionally the ArrayType
    keeps track of metadata that is specific to this datatype such as
    num_classes for an annotated volume or channel names for intensity
    arrays.
    """

    @property
    @abstractmethod
    def interpolatable(self) -> bool:
        """
        This is an abstract method which should be overridden in each of the subclasses
        to determine if an array is interpolatable or not.

        Returns:
            bool: True if the array is interpolatable, False otherwise.
        """
        pass

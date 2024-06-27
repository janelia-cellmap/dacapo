import attr
from typing import Tuple


@attr.s
class ArchitectureConfig:
    """
    A class to represent the base configurations of any architecture. It is used to define the architecture of a neural network model.

    Attributes:
        name : str
            a unique name for the architecture.
    Methods:
        verify()
            validates the given architecture.
    Note:
        The class is abstract and requires to implement the abstract methods.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this architecture. This will be saved so "
            "you and others can find and reuse this task. Keep it short "
            "and avoid special characters."
        }
    )

    def verify(self) -> Tuple[bool, str]:
        """
        A method to validate an architecture configuration.

        Returns:
            Tuple[bool, str]: A tuple of a boolean indicating if the architecture is valid and a message.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> config = ArchitectureConfig("MyModel")
            >>> is_valid, message = config.verify()
            >>> print(is_valid, message)
        Note:
            The method should be implemented in the derived class.
        """
        return True, "No validation for this Architecture"

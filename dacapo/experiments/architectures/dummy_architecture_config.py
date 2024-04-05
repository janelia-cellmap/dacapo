import attr

from .dummy_architecture import DummyArchitecture
from .architecture_config import ArchitectureConfig

from typing import Tuple


@attr.s
class DummyArchitectureConfig(ArchitectureConfig):
    """
    A dummy architecture configuration class used for testing purposes.

    It extends the base class "ArchitectureConfig". This class contains dummy attributes and always
    returns that the configuration is invalid when verified.

    Attributes:
        architecture_type (:obj:`DummyArchitecture`): A class attribute assigning
            the DummyArchitecture class to this configuration.
        num_in_channels (int): The number of input channels. This is a dummy attribute and has no real
            functionality or meaning.
        num_out_channels (int): The number of output channels. This is also a dummy attribute and
            has no real functionality or meaning.
    Methods:
        verify(self) -> Tuple[bool, str]: This method is used to check whether this is a valid architecture configuration.
    Note:
        This class is used to represent a DummyArchitectureConfig object in the system.
    """

    architecture_type = DummyArchitecture

    num_in_channels: int = attr.ib(metadata={"help_text": "Dummy attribute."})

    num_out_channels: int = attr.ib(metadata={"help_text": "Dummy attribute."})

    def verify(self) -> Tuple[bool, str]:
        """
        Verifies the configuration validity.

        Since this is a dummy configuration for testing purposes, this method always returns False
        indicating that the configuration is invalid.

        Returns:
            tuple: A tuple containing a boolean validity flag and a reason message string.
        Raises:
            NotImplementedError: This method is not implemented in this class.
        Examples:
            >>> dummy_architecture_config = DummyArchitectureConfig(num_in_channels=1, num_out_channels=1)
            >>> dummy_architecture_config.verify()
            (False, "This is a DummyArchitectureConfig and is never valid")
        Note:
            This method is used to check whether this is a valid architecture configuration.
        """

        return False, "This is a DummyArchitectureConfig and is never valid"

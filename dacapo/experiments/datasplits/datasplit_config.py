import attr

from typing import Tuple


@attr.s
class DataSplitConfig:
    """
    A class used to create a DataSplit configuration object.

    Attributes:
        name : str
            A name for the datasplit. This name will be saved so it can be found
            and reused easily. It is recommended to keep it short and avoid special
            characters.
    Methods:
        verify() -> Tuple[bool, str]:
            Validates if it is a valid data split configuration.
    Notes:
        This class is used to create a DataSplit configuration object.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this datasplit. This will be saved so "
            "you and others can find and reuse this datasplit. Keep it short "
            "and avoid special characters."
        }
    )

    def verify(self) -> Tuple[bool, str]:
        """
        Validates if the current configuration is a valid data split configuration.

        Returns:
            Tuple[bool, str]
                True if the configuration is valid,
                False otherwise along with respective validation error message.
        Raises:
            NotImplementedError
                If the method is not implemented in the derived class.
        Examples:
            >>> datasplit_config = DataSplitConfig(name="datasplit")
            >>> datasplit_config.verify()
            (True, "No validation for this DataSplit")
        Notes:
            This method is used to validate the configuration of DataSplit.
        """
        return True, "No validation for this DataSplit"

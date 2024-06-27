import attr

from typing import Tuple


@attr.s
class DatasetConfig:
    """
    A class used to define configuration for datasets. This provides the
    framework to create a Dataset instance.

    Attributes:
        name: str (eg: "sample_dataset").
            A unique identifier to name the dataset.
            It aids in easy identification and reusability of this dataset.
            Advised to keep it short and refrain from using special characters.

        weight: int (default=1).
            A numeric value that indicates how frequently this dataset should be
            sampled in comparison to others. Higher the weight, more frequently it
            gets sampled.
    Methods:
        verify:
            Checks and validates the dataset configuration. The specific rules for
            validation need to be defined by the user.
    Notes:
        This class is used to create a configuration object for datasets.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this dataset. This will be saved so you "
            "and others can find and reuse this dataset. Keep it short "
            "and avoid special characters."
        }
    )
    weight: int = attr.ib(
        metadata={
            "help_text": "A weight to indicate this dataset should be sampled from more "
            "heavily"
        },
        default=1,
    )

    def verify(self) -> Tuple[bool, str]:
        """
        Method to verify the dataset configuration.

        Since there is no specific validation logic defined for this DataSet, this
        method will always return True as default reaction and a message stating
        the lack of validation.

        Returns:
            tuple: A tuple of boolean value indicating the check (True or False) and
            message specifying result of validation.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> dataset_config = DatasetConfig(name="sample_dataset")
            >>> dataset_config.verify()
            (True, "No validation for this DataSet")
        Notes:
            This method is used to validate the configuration of the dataset.
        """
        return True, "No validation for this DataSet"

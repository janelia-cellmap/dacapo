import attr
from typing import Tuple

@attr.s
class ArrayConfig:
    """
    A class used to represent array configurations in the application.

    ...

    Attributes
    ----------
    name : str
        A unique name for this array. This will be saved so you 
        and others can find and reuse this array. Keep it short 
        and avoid special characters. 

    Methods
    -------
    verify():
        Checks if a given set of parameters forms a valid array.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this array. This will be saved so you "
            "and others can find and reuse this array. Keep it short "
            "and avoid special characters."
        }
    )

    def verify(self) -> Tuple[bool, str]:
        """
        Function to verify if the array configuration is valid or not.

        Returns
        -------
        Tuple[bool,str]
            Returns a tuple where the first element is a boolean indicating
            the success or failure of the validation process, and the
            second element is a string describing the validation result.
        """
        return True, "No validation for this Array"

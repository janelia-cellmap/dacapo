"""Summary of script: The script is part of the DVID Array Configuration Module 
in the Funkelab DaCapo Python library. It is used to store and verify the basic 
configuration required for a DVID array. The script imports necessary attributes 
and methods from other modules and defines the DVIDArrayConfig class.

The DVIDArrayConfig class inherits the ArrayConfig class and specifies the basic 
attributes for a DVID array. The source attribute holds a tuple of strings and 
the verify method checks the validity of the DVID array.

"""

import attr
from .array_config import ArrayConfig
from .dvid_array import DVIDArray
from typing import Tuple

@attr.s
class DVIDArrayConfig(ArrayConfig):
    """
    DVIDArrayConfig is a configuration class which inherits the properties from 
    ArrayConfig. It outlines the necessary configurations for a DVID array.

    Attributes:
       array_type (DVIDArray): specifies the DVID array type.
       source (Tuple[str]): Holds a tuple of strings describing the source array.

    """
    
    array_type = DVIDArray
    source: Tuple[str, str, str] = attr.ib(metadata={"help_text": "The source strings."})

    def verify(self) -> Tuple[bool, str]:
        """
        Method to verify the validity of the array.
        
        Returns:
           tuple: A tuple determining the validation status and message (True, "No validation for this Array").
           
        """
        return True, "No validation for this Array"

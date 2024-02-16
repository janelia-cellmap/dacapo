"""
Script for SumArrayConfig class which inherits from ArrayConfig. This module is used to configure the Array for the sum 
operation. It's a sub-component of the dacapo library, used for handling sum operations on an Array.

   Attributes:
        array_type: A SumArray object.
        source_array_configs (List[ArrayConfig]): The array of masks from which the union needs to be taken. 
"""

import attr

from .array_config import ArrayConfig
from .sum_array import SumArray

from typing import List


@attr.s
class SumArrayConfig(ArrayConfig):
    """
    This class provides configuration for SumArray. It inherits from ArrayConfig class.
    
    Attributes:
        array_type (SumArray): An attribute to store the SumArray type.
        source_array_configs (List[ArrayConfig]): Lists out the ArrayConfig instances. 
        These configs basically provide information about the source arrays/masks from which the union will be taken.
    """
    array_type = SumArray

    source_array_configs: List[ArrayConfig] = attr.ib(
        metadata={"help_text": "The Array of masks from which to take the union"}
    )
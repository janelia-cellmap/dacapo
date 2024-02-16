```python
import attr

from .array_config import ArrayConfig
from .merge_instances_array import MergeInstancesArray

from typing import List

@attr.s
class MergeInstancesArrayConfig(ArrayConfig):
    """
    A class to represent the configuration of a MergeInstancesArray, inherited from ArrayConfig class.

    Attributes
    ----------
    array_type: class
        Defines the type of array, here it is MergeInstancesArray
    source_array_configs: List[ArrayConfig]
        List of ArrayConfig configurations for source arrays, required for taking union of masks.

    Methods
    -------
    No methods implemented in this class.
    """
    array_type = MergeInstancesArray

    source_array_configs: List[ArrayConfig] = attr.ib(
        metadata={"help_text": "The Array of masks from which to take the union"}
    )
```

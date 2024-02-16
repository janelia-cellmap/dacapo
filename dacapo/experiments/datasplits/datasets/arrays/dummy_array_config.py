```python
import attr

from .array_config import ArrayConfig
from .dummy_array import DummyArray

from typing import Tuple


@attr.s
class DummyArrayConfig(ArrayConfig):
    """
    A dummy array configuration class implemented for the purpose of testing.
    Inherits from the ArrayConfig class. The array_type attribute is set to
    DummyArray by default.

    Attributes:
        array_type: Class object of type DummyArray.
    """
    array_type = DummyArray

    def verify(self) -> Tuple[bool, str]:
        """
        Validate the configuration. As this is a DummyArrayConfig class,
        it is never valid.

        Returns:
            tuple: A tuple containing a boolean indicating the validity
            of the configuration and a string message stating the reason
            of the validation result.
        """
        return False, "This is a DummyArrayConfig and is never valid"
```
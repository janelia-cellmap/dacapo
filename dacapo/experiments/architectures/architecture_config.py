```python
import attr
from typing import Tuple


@attr.s
class ArchitectureConfig:
    """
    A class to represent the base configurations of any architecture.

    Attributes
    ----------
    name : str
        a unique name for the architecture.

    Methods
    -------
    verify()
        validates the given architecture.

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

        Returns
        -------
        bool
            A flag indicating whether the config is valid or not.
        str
            A description of the architecture.
        """
        return True, "No validation for this Architecture"
```
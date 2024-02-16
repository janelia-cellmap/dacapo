Your updated Python source code with added docstrings in Google Style Multi-Line format is:

```Python
import attr

from typing import List


@attr.s(frozen=True)
class PostProcessorParameters:
    """
    Base class for post-processor parameters.

    Attributes:
        id (int): An identifier for the post processor parameters.
    """

    id: int = attr.ib()

    @property
    def parameter_names(self) -> List[str]:
        """
        Getter for parameter names.

        Returns:
            list[str]: A list of parameter names. For this class, it contains only 'id'.
        """
        return ["id"]
# TODO: Add parameter_names to subclasses
```
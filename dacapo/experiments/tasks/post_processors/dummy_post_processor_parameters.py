```python
from .post_processor_parameters import PostProcessorParameters
import attr

@attr.s(frozen=True)
class DummyPostProcessorParameters(PostProcessorParameters):
    """
    A class used to represent the parameters for the dummy post processing step.

    Attributes:
    ----------
    min_size : int
        The minimum size required for the post processing step.

    Args:
    ----------
    min_size : int 
        The minimum size required for the post processing step.

    Returns:
    ----------
        Returns a class instance representing the parameters for the dummy post processing step.

    """

    min_size: int = attr.ib()
```
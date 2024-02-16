```python
from .post_processor_parameters import PostProcessorParameters
import attr

@attr.s(frozen=True)
class ThresholdPostProcessorParameters(PostProcessorParameters):
    """
    A class used to represent the Threshold Post Processor Parameters.

    This class inherits from the PostProcessorParameters class and adds the 
    threshold attribute which holds a float value.

    Attributes
    ----------
    threshold : float
        numerical value at which the thresholding operation is applied, default value is 0.0

    Methods
    -------
    No extra method is added to this class. Only attribute(s) from PostProcessorParameters are inherited.
    """
    
    threshold: float = attr.ib(default=0.0)
```
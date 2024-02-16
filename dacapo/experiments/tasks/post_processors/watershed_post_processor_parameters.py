"""
This module wraps and defines the class WatershedPostProcessorParameters, that it is primarily designed to serve as
a representation of Watershed Post Processor Parameters. The parameters include a bias parameter.

The module uses the PostProcessorParameters class from the post_processor_parameters module to inherit some of its 
attributes. 

Quick note, all the attributes are frozen meaning they can't be modified after initialization. If you try to do so, 
it will throw an error.

Classes:
    WatershedPostProcessorParameters: Defines WatershedPostProcessorParameters with bias as an attribute.
"""

from .post_processor_parameters import PostProcessorParameters
import attr

@attr.s(frozen=True)
class WatershedPostProcessorParameters(PostProcessorParameters):
    """
    A class to represent the Watershed Post Processor Parameters.

    This class inherits the attributes from the class PostProcessorParameters and adds "bias" as an additional 
    attribute.
    
    Attributes
    ----------
    bias : float
        Defines the bias parameter used in watershed post processing. Default value is set to 0.5.
    """
    bias: float = attr.ib(default=0.5)
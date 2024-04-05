from .post_processor_parameters import PostProcessorParameters
import attr
from funlib.geometry import Coordinate


@attr.s(frozen=True)
class WatershedPostProcessorParameters(PostProcessorParameters):
    """
    Parameters for the watershed post-processor.

    Attributes:
        offsets: List of offsets for the watershed transformation.
        threshold: Threshold for the watershed transformation.
        sigma: Sigma for the watershed transformation.
        min_size: Minimum size of the segments.
        bias: Bias for the watershed transformation.
        context: Context for the watershed transformation.
    Examples:
        >>> WatershedPostProcessorParameters(offsets=[(0, 0, 1), (0, 1, 0), (1, 0, 0)], threshold=0.5, sigma=1.0, min_size=100, bias=0.5, context=(32, 32, 32))
    Note:
        This class is used by the ``WatershedPostProcessor`` to define the parameters for the watershed transformation. The offsets are used to define the neighborhood for the watershed transformation.
    """

    bias: float = attr.ib(default=0.5)
    context: Coordinate = attr.ib(default=Coordinate((32, 32, 32)))

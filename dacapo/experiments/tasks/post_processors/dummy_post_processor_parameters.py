from .post_processor_parameters import PostProcessorParameters
import attr


@attr.s(frozen=True)
class DummyPostProcessorParameters(PostProcessorParameters):
    """
    Parameters for the dummy post-processor. The dummy post-processor will set
    the output to 1 if the input is greater than the minimum size, and 0
    otherwise.

    Attributes:
        min_size: The minimum size. If the input is greater than this value, the
            output will be set to 1. Otherwise, the output will be set to 0.
    Methods:
        parameter_names: Get the names of the parameters.
    Note:
        This class is immutable. Once created, the values of its attributes
        cannot be changed.
    """

    min_size: int = attr.ib()

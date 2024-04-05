from .post_processor_parameters import PostProcessorParameters
import attr


@attr.s(frozen=True)
class ArgmaxPostProcessorParameters(PostProcessorParameters):
    """
    Parameters for the argmax post-processor. The argmax post-processor will set
    the output to the index of the maximum value in the input array.

    Methods:
        parameter_names: Get the names of the parameters.
    Note:
        This class is immutable. Once created, the values of its attributes
        cannot be changed.
    """

    pass

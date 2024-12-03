from .post_processor_parameters import PostProcessorParameters
import attr


@attr.s(frozen=True)
class ThresholdPostProcessorParameters(PostProcessorParameters):
    """
    Parameters for the threshold post-processor. The threshold post-processor
    will set the output to 1 if the input is greater than the threshold, and 0
    otherwise.

    Attributes:
        threshold: The threshold value. If the input is greater than this
            value, the output will be set to 1. Otherwise, the output will be
            set to 0.
    Note:
        This class is immutable. Once created, the values of its attributes
        cannot be changed.
    """

    threshold: float = attr.ib(default=0.0)

    def __str__(self):
        return f"threshold_{self.threshold}"

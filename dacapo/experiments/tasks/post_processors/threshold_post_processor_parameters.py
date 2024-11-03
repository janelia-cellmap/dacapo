from .post_processor_parameters import PostProcessorParameters
import attr


@attr.s(frozen=True)
class ThresholdPostProcessorParameters(PostProcessorParameters):
    

    threshold: float = attr.ib(default=0.0)

    def __str__(self):
        return f"threshold_{self.threshold}"

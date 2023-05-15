from .post_processor_parameters import PostProcessorParameters
import attr


@attr.s(frozen=True)
class WatershedPostProcessorParameters(PostProcessorParameters):
    bias: float = attr.ib(default=0.5)

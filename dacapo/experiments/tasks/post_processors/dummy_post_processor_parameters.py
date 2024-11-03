from .post_processor_parameters import PostProcessorParameters
import attr


@attr.s(frozen=True)
class DummyPostProcessorParameters(PostProcessorParameters):
    min_size: int = attr.ib()

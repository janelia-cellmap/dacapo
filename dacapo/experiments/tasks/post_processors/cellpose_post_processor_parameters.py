from .post_processor_parameters import PostProcessorParameters
import attr

# TODO
@attr.s(frozen=True)
class CellposePostProcessorParameters(PostProcessorParameters):
    min_size: int = attr.ib()

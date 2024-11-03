from .post_processor_parameters import PostProcessorParameters
import attr
from funlib.geometry import Coordinate


@attr.s(frozen=True)
class WatershedPostProcessorParameters(PostProcessorParameters):
    

    bias: float = attr.ib(default=0.5)
    context: Coordinate = attr.ib(default=Coordinate((32, 32, 32)))

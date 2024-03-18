import attr
from .cosem_start import CosemStart
from .start_config import StartConfig

@attr.s
class CosemStartConfig(StartConfig):
    """Starter for COSEM pretained models. This is a subclass of `StartConfig` and
    should be used to initialize the model with pretrained weights from a previous
    run.
    """

    start_type = CosemStart
import attr
from .cosem_start import CosemStart
from .start_config import StartConfig


@attr.s
class CosemStartConfig(StartConfig):
    start_type = CosemStart

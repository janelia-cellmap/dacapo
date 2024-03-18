import attr
from .cosem_start import CosemStart


@attr.s
class CosemStartConfig:
    """Starter for COSEM pretained models. This is a subclass of `StartConfig` and
    should be used to initialize the model with pretrained weights from a previous
    run.
    """

    start_type = CosemStart

    name: str = attr.ib(metadata={"help_text": "The COSEM checkpoint name to use."})

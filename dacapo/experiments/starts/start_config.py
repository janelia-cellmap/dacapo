import attr
from .start import Start


@attr.s
class StartConfig:
    

    start_type = Start

    run: str = attr.ib(metadata={"help_text": "The Run to use as a starting point."})
    criterion: str = attr.ib(
        metadata={"help_text": "The criterion for choosing weights from run."}
    )

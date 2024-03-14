import attr
from . import Start


@attr.s
class StartConfig:
    """
    A class to represent the configuration for running tasks.

    Attributes
    ----------
    run : str
        The run to be used as a starting point for tasks.

    criterion : str
        The criterion to be used for choosing weights from run.

    """

    start_type = Start

    run: str = attr.ib(metadata={"help_text": "The Run to use as a starting point."})
    criterion: str = attr.ib(
        metadata={"help_text": "The criterion for choosing weights from run."}
    )

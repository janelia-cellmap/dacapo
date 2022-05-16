import attr

from typing import Tuple


@attr.s
class StartConfig:
    """Base class for task configurations. Each subclass of a `Task` should
    have a corresponding config class derived from `TaskConfig`.
    """

    run: str = attr.ib(metadata={"help_text": "The Run to use as a starting point."})
    criterion: str = attr.ib(
        metadata={"help_text": "The criterion for choosing weights from run."}
    )

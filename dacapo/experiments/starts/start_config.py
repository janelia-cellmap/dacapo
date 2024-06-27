import attr
from .start import Start


@attr.s
class StartConfig:
    """
    A class to represent the configuration for running tasks. This class
    interfaces with the dacapo store to retrieve and load the weights of the
    starter model used for finetuning.

    Attributes:
        run : str
            The run to be used as a starting point for tasks.
        criterion : str
            The criterion to be used for choosing weights from run.
    Methods:
        __init__(start_config)
            Initializes the StartConfig class with specified config to run the
            initialization of weights for a model associated with a specific
            criterion.
    Notes:
        This class is used to represent the configuration for running tasks.
    """

    start_type = Start

    run: str = attr.ib(metadata={"help_text": "The Run to use as a starting point."})
    criterion: str = attr.ib(
        metadata={"help_text": "The criterion for choosing weights from run."}
    )

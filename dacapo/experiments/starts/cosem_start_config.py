import attr
from .cosem_start import CosemStart
from .start_config import StartConfig


@attr.s
class CosemStartConfig(StartConfig):
    """
    Starter for COSEM pretained models. This is a subclass of `StartConfig` and
    should be used to initialize the model with pretrained weights from a previous
    run.

    The weights are loaded from the dacapo store for the specified run. The
    configuration is used to initialize the weights for the model associated with
    a specific criterion.

    Attributes:
        run : str
            The run to be used as a starting point for tasks.
        criterion : str
            The criterion to be used for choosing weights from run.
    Methods:
        __init__(start_config)
            Initializes the CosemStartConfig class with specified config to run the
            initialization of weights for a model associated with a specific
            criterion.
    Examples:
        >>> start_config = CosemStartConfig(run="run_1", criterion="best")
    Notes:
        This class is used to represent the configuration for running tasks.

    """

    start_type = CosemStart

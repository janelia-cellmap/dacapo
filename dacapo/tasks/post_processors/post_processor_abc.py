from abc import ABC, abstractmethod
import time


class PostProcessorABC(ABC):
    """
    This class handles post processing.

    The majority of the complexity of the PostProcessor comes from
    the fact that it must be daisy processing friendly so that it can
    be applied to arbitrarily large volumes.

    Since many common postprocessing pipelines cannot be handled as a
    single daisy task, a PostProcessor is considered to be a series
    of `PostProcessingStep`s
    """

    @abstractmethod
    def tasks(self):
        """
        returns a list of daisy task and a list of parameter dicts defining
        which post processing parameters are used in each task.
        """
        pass

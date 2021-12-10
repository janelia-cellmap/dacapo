from abc import ABC, abstractmethod
from typing import Iterable


class ComputeContext(ABC):

    @property
    @abstractmethod
    def device(self):
        pass

    def train(self, run_name):
        # A helper method to run train in some other context.
        # This can be on a cluster, in a cloud, through bsub,
        # etc.
        # If training should be done locally, return False,
        # else return True.
        return False
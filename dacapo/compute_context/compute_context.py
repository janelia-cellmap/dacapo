from abc import ABC, abstractmethod
from typing import Iterable


class ComputeContext(ABC):

    @property
    @abstractmethod
    def device(self):
        pass
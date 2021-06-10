from abc import ABC, abstractmethod


class LossABC(ABC):
    @abstractmethod
    def instantiate(self):
        pass
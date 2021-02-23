from abc import ABC, abstractmethod


class LossABC(ABC):
    @abstractmethod
    def loss(self):
        pass
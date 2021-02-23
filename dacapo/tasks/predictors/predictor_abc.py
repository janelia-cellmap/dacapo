from abc import ABC, abstractmethod


class PredictorABC(ABC):
    @abstractmethod
    def head(self):
        pass

    @abstractmethod
    def add_target(self):
        pass
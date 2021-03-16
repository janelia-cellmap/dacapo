from abc import ABC, abstractmethod, abstractproperty


class PredictorABC(ABC):
    @abstractmethod
    def head(self):
        pass

    @property
    @abstractmethod
    def fmaps_out(self):
        pass

    @abstractmethod
    def add_target(self):
        pass
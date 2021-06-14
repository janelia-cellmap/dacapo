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

    @property
    def target_fmaps(self):
        # The number of feature maps in the target
        # is generally the same as the fmaps out
        # However, in cases like the 1 hot encoding
        # they can be different.
        return self.fmaps_out
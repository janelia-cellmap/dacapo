from abc import ABC, abstractmethod


class AugmentABC(ABC):
    @abstractmethod
    def node(self, array):
        # get the gunpowder node that performs this augment
        # takes an array. Currently only supports RAW
        pass
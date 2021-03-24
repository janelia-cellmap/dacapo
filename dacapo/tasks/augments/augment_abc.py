from abc import ABC, abstractmethod


class AugmentABC(ABC):
    @abstractmethod
    def node(self):
        # get the gunpowder node that performs this augment
        pass
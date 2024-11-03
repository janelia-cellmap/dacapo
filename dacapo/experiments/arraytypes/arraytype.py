from abc import ABC, abstractmethod


# TODO: Should be read only
class ArrayType(ABC):
    @property
    @abstractmethod
    def interpolatable(self) -> bool:
        pass

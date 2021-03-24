from abc import ABC, abstractmethod


class PostProcessingStepABC(ABC):
    @abstractmethod
    def tasks(self, **kwargs):
        pass

from dacapo.converter import converter

from abc import ABC, abstractmethod


class Algorithm(ABC):
    def verify(self):
        unstructured = converter.unstructure(self)
        structured = converter.structure(unstructured, self.__class__)
        assert self == structured
        return True

    @abstractmethod
    def instance(self, params):
        pass

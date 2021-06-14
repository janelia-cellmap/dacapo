import attr

from dacapo.converter import converter
from .algorithms import AnyAlgorithm


@attr.s
class Optimizer:
    name: str = attr.ib(
        metadata={"help_text": "Name of optimizer for easy search and reuse"}
    )
    algorithm: AnyAlgorithm = attr.ib(
        metadata={"help_text": "The algorithm to use for optimization"}
    )
    batch_size: int = attr.ib(default=2)

    def instance(self, params):
        return self.algorithm.instance(params)

    def verify(self):
        unstructured = converter.unstructure(self)
        structured = converter.structure(unstructured, self.__class__)
        assert self == structured
        return self.algorithm.verify()

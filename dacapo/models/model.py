from .architectures import AnyArchitecture
from dacapo.converter import converter

import attr


@attr.s
class Model:
    name: str = attr.ib(
        metadata={"help_text": "Name of your model for easy search and reuse"}
    )
    architecture: AnyArchitecture = attr.ib(metadata={"help_text": "The type of model"})

    def verify(self):
        unstructured = converter.unstructure(self)
        structured = converter.structure(unstructured, self.__class__)
        assert self == structured
        self.architecture.verify()

    def __getattr__(self, name):
        return getattr(self.architecture, name)
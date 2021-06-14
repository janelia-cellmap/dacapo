from .argmax import ArgMax
from .watershed import Watershed
from dacapo.converter import converter

from typing import Union

AnyPostProcessor = Union[ArgMax, Watershed]

converter.register_unstructure_hook(
    AnyPostProcessor,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    AnyPostProcessor,
    lambda o, _: converter.structure(o, eval(o["__type__"])),
)
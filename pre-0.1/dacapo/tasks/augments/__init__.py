from .intensity_augment import IntensityAugment
from .simple_augment import SimpleAugment
from dacapo.converter import converter

from typing import Union

AnyAugment = Union[SimpleAugment, IntensityAugment]

converter.register_unstructure_hook(
    AnyAugment,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    AnyAugment,
    lambda o, _: converter.structure(o, eval(o.pop("__type__"))),
)
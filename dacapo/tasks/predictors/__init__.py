from .affinities import Affinities  # noqa
from .one_hot_labels import OneHotLabels  # noqa
from .lsd import LSD  # noqa
from dacapo.converter import converter

from typing import Union

AnyPredictor = Union[Affinities, OneHotLabels, LSD]

converter.register_unstructure_hook(
    AnyPredictor,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    AnyPredictor,
    lambda o, _: converter.structure(o, eval(o["__type__"])),
)

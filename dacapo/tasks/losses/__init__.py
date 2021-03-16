from .cross_entropy_loss import CrossEntropyLoss
from .mse_loss import MSELoss
from dacapo.converter import converter

from typing import Union, get_args

AnyLoss = Union[MSELoss, CrossEntropyLoss]

converter.register_unstructure_hook(
    AnyLoss,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    AnyLoss,
    lambda o, _: converter.structure(o, eval(o.pop("__type__"))),
)
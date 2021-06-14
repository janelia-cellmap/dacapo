from .unet import UNet
from .vggnet import VGGNet
from dacapo.converter import converter

from typing import Union, get_args

AnyArchitecture = Union[UNet, VGGNet]

converter.register_unstructure_hook(
    AnyArchitecture,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    AnyArchitecture,
    lambda o, _: converter.structure(o, eval(o["__type__"])),
)

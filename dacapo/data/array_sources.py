from .zarr_source import ZarrSource
from .bossdb import BossDBSource
from .rasterize_source import RasterizeSource
from dacapo.converter import converter

from typing import Union

AnyArraySource = Union[ZarrSource, BossDBSource, RasterizeSource]

converter.register_unstructure_hook(
    AnyArraySource,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    AnyArraySource,
    lambda o, _: converter.structure(o, eval(o["__type__"])),
)

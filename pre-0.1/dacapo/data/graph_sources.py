from .csv_source import CSVSource
from .nxgraph import NXGraphSource
from dacapo.converter import converter


from typing import Union

AnyGraphSource = Union[CSVSource, NXGraphSource]

converter.register_unstructure_hook(
    AnyGraphSource,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    AnyGraphSource,
    lambda o, _: converter.structure(o, eval(o["__type__"])),
)

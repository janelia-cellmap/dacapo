import attr

from .array_sources import AnyArraySource
from .graph_sources import AnyGraphSource
from dacapo.converter import converter

from typing import List, Union


@attr.s
class ArrayDataSource:
    """
    The class representing a single data key. i.e. `raw` will point to
    the `raw` array group which can consist of any number of array
    sources that all provide a `raw` dataset
    """

    arrays: List[AnyArraySource] = attr.ib(factory=list)

    def __getattr__(self, name):
        value = None
        for array in self.arrays:
            if value is None:
                value = getattr(array, name)
            else:
                assert value == getattr(array, name), (
                    f"array sources have multiple values: ({value, getattr(array, name)}) "
                    f"associated with attribute: {name}"
                )
        return value

    def add_source(self, source: AnyArraySource):
        if isinstance(source, ArrayDataSource):
            raise Exception()
        self.arrays.append(source)

    def get_sources(self, array_key, override_spec=None):
        print(self.arrays)
        return [source.get_source(array_key, override_spec) for source in self.arrays]


@attr.s
class GraphDataSource:
    graphs: List[AnyGraphSource] = attr.ib(factory=list)

    def __getattr__(self, name):
        value = None
        for graph in self.graphs:
            if value is None:
                value = getattr(graph, name)
            else:
                assert value == getattr(graph, name)
        return value

    def add_source(self, source: AnyGraphSource):
        self.graphs.append(source)

    def get_sources(self, graph_key, override_spec=None):
        return [source.get_source(graph_key, override_spec) for source in self.graphs]


DataSource = Union[ArrayDataSource, GraphDataSource]

converter.register_unstructure_hook(
    List[AnyArraySource],
    lambda o: [{**converter.unstructure(e, unstructure_as=AnyArraySource)} for e in o],
)
converter.register_unstructure_hook(
    List[AnyGraphSource],
    lambda o: [{**converter.unstructure(e, unstructure_as=AnyGraphSource)} for e in o],
)
converter.register_unstructure_hook(
    DataSource,
    lambda o: {"__type__": type(o).__name__, **converter.unstructure(o)},
)
converter.register_structure_hook(
    DataSource,
    lambda o, _: converter.structure(o, eval(o["__type__"])),
)

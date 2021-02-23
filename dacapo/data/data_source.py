from typing import List, Union
import attr

from .array_sources import AnyArraySource
from .graph_sources import AnyGraphSource


@attr.s
class ArrayGroup:
    """
    The class representing a single data key. i.e. `raw` will point to
    the `raw` array group which can consist of any number of array
    sources that all provide a `raw` dataset
    """

    arrays: List[AnyArraySource] = attr.ib()


@attr.s
class GraphGroup:
    graphs: List[AnyGraphSource] = attr.ib()


DataSource = Union[ArrayGroup, GraphGroup]

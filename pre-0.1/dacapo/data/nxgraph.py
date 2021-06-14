import attr

from dacapo.gp import NXSource

from pathlib import Path


@attr.s
class NXGraphSource:
    """
    The class representing a csv data source
    """

    filename: Path = attr.ib()

    def get_source(self, graph, overwrite_spec=None):
        return NXSource(graph, self.filename)

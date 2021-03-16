import attr
import gunpowder as gp

from pathlib import Path
from typing import Optional


@attr.s
class CSVSource:
    """
    The class representing a csv data source
    """

    filename: Path = attr.ib()
    ndims: Optional[int] = attr.ib(default=None)
    id_dim: Optional[int] = attr.ib(default=None)

    def get_source(self, graph, overwrite_spec=None):
        return gp.CsvPointsSource(
            self.filename, graph, id_dim=self.id_dim, ndims=self.ndims
        )

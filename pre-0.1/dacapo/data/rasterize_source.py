from gunpowder import (
    ArrayKey,
    GraphKey,
    ArraySpec,
    RasterizeGraph,
    RasterizationSettings,
)
from funlib.geometry import Coordinate, Roi
import attr

from .graph_sources import AnyGraphSource

from typing import List, Optional


@attr.s
class RasterizationSetting:

    radius: float = attr.ib()
    mode: str = attr.ib(default="ball")
    mask: Optional[str] = attr.ib(default=None)
    inner_radius_fraction: Optional[float] = attr.ib(default=None)
    fg_value: int = attr.ib(default=1)
    bg_value: int = attr.ib(default=0)
    edges: bool = attr.ib(default=True)
    color_attr: Optional[str] = attr.ib(default=None)

    def instantiate(self):
        return RasterizationSettings(
            radius=self.radius,
            mode=self.mode,
            mask=self.mask,
            inner_radius_fraction=self.inner_radius_fraction,
            fg_value=self.fg_value,
            bg_value=self.bg_value,
            edges=self.edges,
            color_attr=self.color_attr,
        )


@attr.s
class RasterizeSource:
    """
    The class representing a single zarr dataset
    """

    graph_source: AnyGraphSource = attr.ib()

    rasterization_settings: RasterizationSetting = attr.ib()

    axes: List[str] = attr.ib()
    voxel_size: Coordinate = attr.ib()

    def get_source(self, array, overwrite_spec=None):
        spec = overwrite_spec if overwrite_spec is not None else ArraySpec()
        spec.voxel_size = self.voxel_size
        temp = GraphKey(f"{array}-GRAPH")
        graph_source = self.graph_source.get_source(temp)
        rasterized_source = graph_source + RasterizeGraph(
            temp,
            array,
            array_spec=overwrite_spec,
            settings=self.rasterization_settings.instantiate(),
        )
        return rasterized_source

    # if using a rasterized source as ground truth, we need to know number of classes
    # maybe that shouldn't be stored here though since it won't always be relevant
    # to the rasterization class
    @property
    def num_classes(self):
        return 2

    @property
    def dims(self):
        return self.voxel_size.dims

    @property
    def roi(self):
        return Roi(Coordinate((None,) * self.dims), Coordinate((None,) * self.dims))

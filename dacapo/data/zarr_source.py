import gunpowder as gp
from funlib.geometry import Coordinate
import zarr

import logging

from typing import List, Optional
from pathlib import Path
import attr


@attr.s
class ZarrSource:
    """
    The class representing a single zarr dataset
    """

    filename: Path = attr.ib()
    ds_name: str = attr.ib()

    axes: Optional[List[str]] = attr.ib(default=None)
    voxel_size: Optional[Coordinate] = attr.ib(default=None)
    offset: Optional[Coordinate] = attr.ib(default=None)
    # roi: Optional[Roi] = attr.ib(default=None)

    num_classes: Optional[int] = attr.ib(default=None)

    def __attrs_post_init__(self):
        zarr_container = zarr.open(str(self.filename))
        ds = zarr_container[self.ds_name]

        assert "axes" in ds.attrs, (
            "Dacapo expects zarr arrays to come with axis labels. "
            "Note that label 'c' is reserved for the (optional) channel dimension. "
            "Label 's' is reserved for the (optional) sample dimension. "
            "Any other label will be treated as a spatial dimension."
        )

        # calculate useful metadata:
        # axes
        axes = ds.attrs["axes"]
        axis_map = {d: a for d, a in enumerate(ds.attrs["axes"])}
        inv_axes = {a: d for d, a in axis_map.items()}
        channel_dim = inv_axes.get("c")
        spatial_axes = sorted(
            [i for i in range(len(axes)) if i != channel_dim]
        )
        spatial_dims = len(spatial_axes)

        # optional fields
        if "resolution" in ds.attrs:
            voxel_size = Coordinate(ds.attrs["resolution"])
        else:
            voxel_size = Coordinate([1] * spatial_dims)
        if "offset" in ds.attrs:
            offset = Coordinate(ds.attrs["offset"])
        else:
            offset = Coordinate([0] * spatial_dims)

        # data attributes
        shape = ds.shape
        spatial_shape = Coordinate([shape[i] for i in spatial_axes])
        roi = gp.Roi(offset, spatial_shape * voxel_size)
        num_channels = shape[channel_dim] if channel_dim is not None else 1

        for attribute, value in [
            ("axes", axes),
            ("voxel_size", voxel_size),
            ("offset", offset),
        ]:
            self._validate_or_set(attribute, value)

        self.shape = shape
        self.channel_dim = channel_dim
        self.spatial_axes = spatial_axes
        self.spatial_dims = spatial_dims
        self.spatial_shape = spatial_shape
        self.roi = roi
        self.num_channels = num_channels

    def _validate_or_set(self, attribute, value):
        if getattr(self, attribute) is None:
            setattr(self, attribute, value)
        else:
            assert getattr(self, attribute) == value, (
                f"provided {attribute} ({getattr(self, attribute)}) does "
                f"not match calculated {attribute} ({value})"
            )

    def get_source(self, array, overwrite_spec=None):

        if overwrite_spec:
            return gp.ZarrSource(
                str(self.filename),
                {array: self.ds_name},
                array_specs={array: overwrite_spec},
            )
        else:
            return gp.ZarrSource(str(self.filename), {array: self.ds_name})

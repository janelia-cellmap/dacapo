import gunpowder as gp
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

    shape: Optional[gp.Coordinate] = attr.ib(default=None)
    axes: Optional[List[str]] = attr.ib(default=None)
    voxel_size: Optional[gp.Coordinate] = attr.ib(default=None)
    offset: Optional[gp.Coordinate] = attr.ib(default=None)
    channel_dim: Optional[int] = attr.ib(default=None)
    sample_dim: Optional[int] = attr.ib(default=None)
    spatial_axes: Optional[List[str]] = attr.ib(default=None)
    spatial_dims: Optional[int] = attr.ib(default=None)
    spatial_shape: Optional[gp.Coordinate] = attr.ib(default=None)
    roi: Optional[gp.Roi] = attr.ib(default=None)
    num_channels: Optional[int] = attr.ib(default=None)
    num_samples: Optional[int] = attr.ib(default=None)

    def __attrs_post_init__(self):
        zarr_container = zarr.open(self.filename)
        ds = zarr_container[self.ds_name]

        assert "axes" in ds.attrs, (
            "Dacapo expects zarr arrays to come with axis labels. "
            "Note that label 'c' is reserved for the (optional) channel dimension. "
            "Label 's' is reserved for the (optional) sample dimension. "
            "Any other label will be treated as a spatial dimension."
        )

        # calculate useful metadata:
        # axes
        axes = {d: a for d, a in enumerate(ds.attrs["axes"])}
        inv_axes = {a: d for d, a in axes.items()}
        channel_dim = inv_axes.get("c")
        sample_dim = inv_axes.get("s")
        spatial_axes = sorted(
            [i for i in axes.keys() if i != channel_dim and i != sample_dim]
        )
        spatial_dims = len(spatial_axes)

        # optional fields
        if "resolution" in ds.attrs:
            voxel_size = list(ds.attrs["resolution"])
        else:
            voxel_size = [1 for i in spatial_axes]
        if "offset" in ds.attrs:
            offset = list(ds.attrs["offset"])
        else:
            offset = [0 for i in spatial_axes]

        # data attributes
        shape = ds.shape
        spatial_shape = [shape[i] for i in spatial_axes]
        roi = gp.Roi(offset, spatial_shape * voxel_size)
        num_channels = shape[channel_dim] if channel_dim is not None else 0
        num_samples = shape[sample_dim] if sample_dim is not None else 0

        for attribute, value in [
            ("shape", shape),
            ("axes", axes),
            ("voxel_size", voxel_size),
            ("offset", offset),
            ("channel_dim", channel_dim),
            ("sample_dim", sample_dim),
            ("spatial_axes", spatial_axes),
            ("spatial_dims", spatial_dims),
            ("spatial_shape", spatial_shape),
            ("roi", roi),
            ("num_channels", num_channels),
            ("num_samples", num_samples),
        ]:
            self._validate_or_set(attribute, value)

    def _validate_or_set(self, attribute, value):
        if getattr(self, attribute) is not None:
            setattr(self, attribute, value)
        else:
            assert getattr(self, attribute) == value, (
                f"provided {attribute} ({getattr(self, attribute)}) does "
                f"not match calculated {attribute} ({value})"
            )

    def get_source(self, array, overwrite_spec=None):

        if overwrite_spec:
            return gp.ZarrSource(
                self.filename,
                {array: self.ds_name},
                array_specs={array: overwrite_spec},
            )
        else:
            return gp.ZarrSource(self.filename, {array: self.ds_name})

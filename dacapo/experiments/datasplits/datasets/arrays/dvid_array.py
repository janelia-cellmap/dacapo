from .array import Array
from dacapo.ext import NoSuchModule

try:
    from neuclease.dvid import fetch_info, fetch_labelmap_voxels, fetch_raw
except ImportError:
    fetch_info = NoSuchModule("neuclease.dvid.fetch_info")
    fetch_labelmap_voxels = NoSuchModule("neuclease.dvid.fetch_labelmap_voxels")

from funlib.geometry import Coordinate, Roi
import funlib.persistence

import neuroglancer

import lazy_property
import numpy as np

import logging
from typing import Dict, Tuple, Any, Optional, List

logger = logging.getLogger(__name__)


class DVIDArray(Array):
    """This is a DVID array"""

    def __init__(self, array_config):
        super().__init__()
        self.name: str = array_config.name
        self.source: tuple[str, str, str] = array_config.source

    def __str__(self):
        return f"DVIDArray({self.source})"

    def __repr__(self):
        return f"DVIDArray({self.source})"

    @lazy_property.LazyProperty
    def attrs(self):
        return fetch_info(*self.source)

    @property
    def axes(self):
        return ["c", "z", "y", "x"][-self.dims :]

    @property
    def dims(self) -> int:
        return self.voxel_size.dims

    @lazy_property.LazyProperty
    def _daisy_array(self) -> funlib.persistence.Array:
        raise NotImplementedError()

    @lazy_property.LazyProperty
    def voxel_size(self) -> Coordinate:
        return Coordinate(self.attrs["Extended"]["VoxelSize"])

    @lazy_property.LazyProperty
    def roi(self) -> Roi:
        return Roi(
            Coordinate(self.attrs["Extents"]["MinPoint"]) * self.voxel_size,
            Coordinate(self.attrs["Extents"]["MaxPoint"]) * self.voxel_size,
        )

    @property
    def writable(self) -> bool:
        return False

    @property
    def dtype(self) -> Any:
        return np.dtype(self.attrs["Extended"]["Values"][0]["DataType"])

    @property
    def num_channels(self) -> Optional[int]:
        return None

    @property
    def spatial_axes(self) -> List[str]:
        return [ax for ax in self.axes if ax not in set(["c", "b"])]

    @property
    def data(self) -> Any:
        raise NotImplementedError()

    def __getitem__(self, roi: Roi) -> np.ndarray[Any, Any]:
        box = np.array(
            (roi.offset / self.voxel_size, (roi.offset + roi.shape) / self.voxel_size)
        )
        if self.source[2] == "grayscale":
            data = fetch_raw(*self.source, box)
        elif self.source[2] == "segmentation":
            data = fetch_labelmap_voxels(*self.source, box)
        else:
            raise Exception(self.source)
        return data

    def _can_neuroglance(self) -> bool:
        return True

    def _neuroglancer_source(self):
        raise NotImplementedError()

    def _neuroglancer_layer(self) -> Tuple[neuroglancer.ImageLayer, Dict[str, Any]]:
        raise NotImplementedError()

    def _transform_matrix(self):
        raise NotImplementedError()

    def _output_dimensions(self) -> Dict[str, Tuple[float, str]]:
        raise NotImplementedError()

    def _source_name(self) -> str:
        raise NotImplementedError()

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        raise NotImplementedError()

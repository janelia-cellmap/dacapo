from .array import Array
from dacapo import Options
from funlib.persistence import open_ds
from funlib.geometry import Coordinate, Roi
import funlib.persistence

import neuroglancer

import lazy_property
import numpy as np
import zarr

from collections import OrderedDict
import logging
from pathlib import Path
import json
from typing import Dict, Tuple, Any, Optional, List

logger = logging.getLogger(__name__)


class ZarrArray(Array):
    """This is a zarr array"""

    def __init__(self, array_config):
        super().__init__()
        self.name = array_config.name
        self.file_name = array_config.file_name
        self.dataset = array_config.dataset

        self._attributes = self.data.attrs
        self._axes = array_config._axes
        self.snap_to_grid = array_config.snap_to_grid

    def __str__(self):
        return f"ZarrArray({self.file_name}, {self.dataset})"

    def __repr__(self):
        return f"ZarrArray({self.file_name}, {self.dataset})"

    @property
    def attrs(self):
        return self.data.attrs

    @property
    def axes(self):
        if self._axes is not None:
            return self._axes
        try:
            return self._attributes["axes"]
        except KeyError:
            logger.debug(
                "DaCapo expects Zarr datasets to have an 'axes' attribute!\n"
                f"Zarr {self.file_name} and dataset {self.dataset} has attributes: {list(self._attributes.items())}\n"
                f"Using default {['s', 'c', 'z', 'y', 'x'][-self.dims::]}",
            )
            return ["s", "c", "z", "y", "x"][-self.dims : :]

    @property
    def dims(self) -> int:
        return self.voxel_size.dims

    @lazy_property.LazyProperty
    def _daisy_array(self) -> funlib.persistence.Array:
        return funlib.persistence.open_ds(f"{self.file_name}", self.dataset)

    @lazy_property.LazyProperty
    def voxel_size(self) -> Coordinate:
        return self._daisy_array.voxel_size

    @lazy_property.LazyProperty
    def roi(self) -> Roi:
        if self.snap_to_grid is not None:
            return self._daisy_array.roi.snap_to_grid(self.snap_to_grid, mode="shrink")
        else:
            return self._daisy_array.roi

    @property
    def writable(self) -> bool:
        return True

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    @property
    def num_channels(self) -> Optional[int]:
        return None if "c" not in self.axes else self.data.shape[self.axes.index("c")]

    @property
    def spatial_axes(self) -> List[str]:
        return [ax for ax in self.axes if ax not in set(["c", "b"])]

    @property
    def data(self) -> Any:
        zarr_container = zarr.open(str(self.file_name))
        return zarr_container[self.dataset]

    def __getitem__(self, roi: Roi) -> np.ndarray:
        data: np.ndarray = funlib.persistence.Array(
            self.data, self.roi, self.voxel_size
        ).to_ndarray(roi=roi)
        return data

    def __setitem__(self, roi: Roi, value: np.ndarray):
        funlib.persistence.Array(self.data, self.roi, self.voxel_size)[roi] = value

    @classmethod
    def create_from_array_identifier(
        cls,
        array_identifier,
        axes,
        roi,
        num_channels,
        voxel_size,
        dtype,
        write_size=None,
        name=None,
        overwrite=False,
    ):
        """
        Create a new ZarrArray given an array identifier. It is assumed that
        this array_identifier points to a dataset that does not yet exist
        """
        if write_size is None:
            # total storage per block is approx c*x*y*z*dtype_size
            # appropriate block size about 5MB.
            axis_length = (
                (
                    1024**2
                    * 5
                    / (num_channels if num_channels is not None else 1)
                    / np.dtype(dtype).itemsize
                )
                ** (1 / voxel_size.dims)
            ) // 1
            write_size = Coordinate((axis_length,) * voxel_size.dims) * voxel_size
        write_size = Coordinate((min(a, b) for a, b in zip(write_size, roi.shape)))
        zarr_container = zarr.open(array_identifier.container, "a")
        if num_channels is None or num_channels == 1:
            axes = [axis for axis in axes if "c" not in axis]
            num_channels = None
        else:
            axes = ["c"] + [axis for axis in axes if "c" not in axis]
        try:
            funlib.persistence.prepare_ds(
                f"{array_identifier.container}",
                array_identifier.dataset,
                roi,
                voxel_size,
                dtype,
                num_channels=num_channels,
                write_size=write_size,
                delete=overwrite,
                force_exact_write_size=True,
            )
            zarr_dataset = zarr_container[array_identifier.dataset]
            if array_identifier.container.name.endswith("n5"):
                zarr_dataset.attrs["offset"] = roi.offset[::-1]
                zarr_dataset.attrs["resolution"] = voxel_size[::-1]
                zarr_dataset.attrs["axes"] = axes[::-1]
                # to make display right in neuroglancer: TODO ADD CHANNELS
                zarr_dataset.attrs["dimension_units"] = [
                    f"{size} nm" for size in voxel_size[::-1]
                ]
                zarr_dataset.attrs["_ARRAY_DIMENSIONS"] = [
                    a if a != "c" else "c^" for a in axes[::-1]
                ]
            else:
                zarr_dataset.attrs["offset"] = roi.offset
                zarr_dataset.attrs["resolution"] = voxel_size
                zarr_dataset.attrs["axes"] = axes
                # to make display right in neuroglancer: TODO ADD CHANNELS
                zarr_dataset.attrs["dimension_units"] = [
                    f"{size} nm" for size in voxel_size
                ]
                zarr_dataset.attrs["_ARRAY_DIMENSIONS"] = [
                    a if a != "c" else "c^" for a in axes
                ]
            if "c" in axes:
                if axes.index("c") == 0:
                    zarr_dataset.attrs["dimension_units"] = [
                        str(num_channels)
                    ] + zarr_dataset.attrs["dimension_units"]
                else:
                    zarr_dataset.attrs["dimension_units"] = zarr_dataset.attrs[
                        "dimension_units"
                    ] + [str(num_channels)]
        except zarr.errors.ContainsArrayError:
            zarr_dataset = zarr_container[array_identifier.dataset]
            assert (
                tuple(zarr_dataset.attrs["offset"]) == roi.offset
            ), f"{zarr_dataset.attrs['offset']}, {roi.offset}"
            assert (
                tuple(zarr_dataset.attrs["resolution"]) == voxel_size
            ), f"{zarr_dataset.attrs['resolution']}, {voxel_size}"
            assert tuple(zarr_dataset.attrs["axes"]) == tuple(
                axes
            ), f"{zarr_dataset.attrs['axes']}, {axes}"
            assert (
                zarr_dataset.shape
                == ((num_channels,) if num_channels is not None else ())
                + roi.shape / voxel_size
            ), f"{zarr_dataset.shape}, {((num_channels,) if num_channels is not None else ()) + roi.shape / voxel_size}"
            zarr_dataset[:] = np.zeros(zarr_dataset.shape, dtype)

        zarr_array = cls.__new__(cls)
        zarr_array.file_name = array_identifier.container
        zarr_array.dataset = array_identifier.dataset
        zarr_array._axes = None
        zarr_array._attributes = zarr_array.data.attrs
        zarr_array.snap_to_grid = None
        return zarr_array

    @classmethod
    def open_from_array_identifier(cls, array_identifier, name=""):
        zarr_array = cls.__new__(cls)
        zarr_array.name = name
        zarr_array.file_name = array_identifier.container
        zarr_array.dataset = array_identifier.dataset
        zarr_array._axes = None
        zarr_array._attributes = zarr_array.data.attrs
        zarr_array.snap_to_grid = None
        return zarr_array

    def _can_neuroglance(self) -> bool:
        return True

    def _neuroglancer_source(self):
        d = open_ds(str(self.file_name), self.dataset)
        return neuroglancer.LocalVolume(
            data=d.data,
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units=["nm", "nm", "nm"],
                scales=self.voxel_size,
            ),
            voxel_offset=self.roi.get_begin() / self.voxel_size,
        )

    def _neuroglancer_layer(self) -> Tuple[neuroglancer.ImageLayer, Dict[str, Any]]:
        layer = neuroglancer.ImageLayer(source=self._neuroglancer_source())
        return layer

    def _transform_matrix(self):
        is_zarr = self.file_name.name.endswith(".zarr")
        if is_zarr:
            offset = self.roi.offset
            voxel_size = self.voxel_size
            matrix = [
                [0] * (self.dims - i - 1) + [1e-9 * vox] + [0] * i + [off / vox]
                for i, (vox, off) in enumerate(zip(voxel_size[::-1], offset[::-1]))
            ]
            if "c" in self.axes:
                matrix = [[1] + [0] * (self.dims + 1)] + [[0] + row for row in matrix]
            return matrix
        else:
            offset = self.roi.offset[::-1]
            voxel_size = self.voxel_size[::-1]
            matrix = [
                [0] * (self.dims - i - 1) + [1] + [0] * i + [off]
                for i, (vox, off) in enumerate(zip(voxel_size[::-1], offset[::-1]))
            ]
            if "c" in self.axes:
                matrix = [[1] + [0] * (self.dims + 1)] + [[0] + row for row in matrix]
            return matrix
            return [[0] * i + [1] + [0] * (self.dims - i) for i in range(self.dims)]

    def _output_dimensions(self) -> Dict[str, Tuple[float, str]]:
        is_zarr = self.file_name.name.endswith(".zarr")
        if is_zarr:
            spatial_dimensions = OrderedDict()
            if "c" in self.axes:
                spatial_dimensions["c^"] = (1.0, "")
            for dim, vox in zip(self.spatial_axes[::-1], self.voxel_size[::-1]):
                spatial_dimensions[dim] = (vox * 1e-9, "m")
            return spatial_dimensions
        else:
            return {
                dim: (1e-9, "m")
                for dim, vox in zip(self.spatial_axes[::-1], self.voxel_size[::-1])
            }

    def _source_name(self) -> str:
        return self.name

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        dataset = zarr.open(self.file_name, mode="a")[self.dataset]
        for k, v in metadata.items():
            dataset.attrs[k] = v

from os import symlink
from .array import Array
from dacapo import Options

from funlib.geometry import Coordinate, Roi
import daisy

import neuroglancer

import lazy_property
import numpy as np
import zarr

from collections import OrderedDict
import logging
from pathlib import Path
import json

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

    @property
    def attrs(self):
        return self.data.attrs

    @property
    def axes(self):
        if self._axes is not None:
            return self._axes
        try:
            return self._attributes["axes"]
        except KeyError as e:
            logger.debug(
                "DaCapo expects Zarr datasets to have an 'axes' attribute!\n"
                f"Zarr {self.file_name} and dataset {self.dataset} has attributes: {list(self._attributes.items())}\n"
                f"Using default {['t', 'z', 'y', 'x'][-self.dims::]}",
            )
            return ["t", "z", "y", "x"][-self.dims : :]

    @property
    def dims(self) -> int:
        return self.voxel_size.dims

    @lazy_property.LazyProperty
    def _daisy_array(self) -> daisy.Array:
        return daisy.open_ds(f"{self.file_name}", self.dataset)

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
    def writable(self):
        return True

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def num_channels(self):
        return None if "c" not in self.axes else self.data.shape[self.axes.index("c")]

    @property
    def spatial_axes(self):
        return [ax for ax in self.axes if ax not in set(["c", "b"])]

    @property
    def data(self):
        zarr_container = zarr.open(str(self.file_name))
        return zarr_container[self.dataset]

    def __getitem__(self, roi: Roi) -> np.ndarray:
        return daisy.Array(self.data, self.roi, self.voxel_size).to_ndarray(roi=roi)

    def __setitem__(self, roi: Roi, value: np.ndarray):
        daisy.Array(self.data, self.roi, self.voxel_size)[roi] = value

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
        try:
            daisy.prepare_ds(
                f"{array_identifier.container}",
                array_identifier.dataset,
                roi,
                voxel_size,
                dtype,
                num_channels=num_channels,
                write_size=write_size,
            )
            zarr_dataset = zarr_container[array_identifier.dataset]
            zarr_dataset.attrs["offset"] = roi.offset
            zarr_dataset.attrs["resolution"] = voxel_size
            zarr_dataset.attrs["axes"] = axes
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

    def _can_neuroglance(self):
        return True

    def _neuroglancer_source(self):
        source_type = "n5" if self.file_name.name.endswith(".n5") else "zarr"
        options = Options.instance()
        base_dir = Path(options.runs_base_dir)
        try:
            relpath = self.file_name.relative_to(base_dir)
        except ValueError:
            relpath = self.file_name.name
        symlink_path = f"data_symlinks/{relpath}"

        # Check if data is symlinked to a servable location
        if not (base_dir / symlink_path).exists():
            if not (base_dir / symlink_path).parent.exists():
                (base_dir / symlink_path).parent.mkdir()
            (base_dir / symlink_path).symlink_to(Path(self.file_name))

        dataset = self.dataset
        parent_attributes_path = (
            base_dir / symlink_path / self.dataset
        ).parent / "attributes.json"
        if parent_attributes_path.exists():
            dataset_parent_attributes = json.loads(
                open(
                    (base_dir / symlink_path / self.dataset).parent / "attributes.json",
                    "r",
                ).read()
            )
            if "scales" in dataset_parent_attributes:
                dataset = "/".join(self.dataset.split("/")[:-1])

        source = {
            "url": f"{source_type}://{options.file_server}/{symlink_path}/{dataset}",
            "transform": {
                "matrix": self._transform_matrix(),
                "outputDimensions": self._output_dimensions(),
            },
        }
        return source

    def _neuroglancer_layer(self):
        # Generates an Image layer. May not be correct if this crop contains a segmentation

        layer = neuroglancer.ImageLayer(source=self._neuroglancer_source())
        kwargs = {
            "visible": False,
            "blend": "additive",
        }
        return layer, kwargs

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
            return [[0] * i + [1] + [0] * (self.dims - i) for i in range(self.dims)]

    def _output_dimensions(self):
        is_zarr = self.file_name.name.endswith(".zarr")
        if is_zarr:
            spatial_dimensions = OrderedDict()
            if "c" in self.axes:
                spatial_dimensions["c^"] = [1, ""]
            for dim, vox in zip(self.spatial_axes[::-1], self.voxel_size[::-1]):
                spatial_dimensions[dim] = [vox * 1e-9, "m"]
            return spatial_dimensions
        else:
            return {
                dim: [vox * 1e-9, "m"]
                for dim, vox in zip(self.spatial_axes, self.voxel_size)
            }

    def _source_name(self):
        return self.name

    def add_metadata(self, metadata):
        dataset = zarr.open(self.file_name, mode="a")[self.dataset]
        for k, v in metadata.items():
            dataset.attrs[k] = v

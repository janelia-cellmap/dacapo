from .array import Array

from funlib.geometry import Coordinate, Roi
import daisy

import numpy as np
import zarr


class ZarrArray(Array):
    """This is a zarr array"""

    def __init__(self, array_config):
        super().__init__()
        self.file_name = array_config.file_name
        self.dataset = array_config.dataset

        self._attributes = self.data.attrs

    @property
    def axes(self):
        try:
            return self._attributes["axes"]
        except KeyError as e:
            raise KeyError(
                "DaCapo expects Zarr datasets to have an 'axes' attribute! ",
                f"Zarr {self.file_name} and dataset {self.dataset} has attributes: {self._attributes}",
            ) from e

    @property
    def dims(self) -> int:
        return self.voxel_size.dims

    @property
    def voxel_size(self) -> Coordinate:
        try:
            return Coordinate(self._attributes["resolution"])
        except KeyError as e:
            raise KeyError(
                "DaCapo expects Zarr datasets to have an 'resolution' attribute! ",
                f"Zarr {self.file_name} and dataset {self.dataset} has attributes: {self._attributes}",
            ) from e

    @property
    def roi(self) -> Roi:
        try:
            offset = self._attributes["offset"]
            shape = Coordinate(self.data.shape[-self.dims :]) * self.voxel_size
            return Roi(offset, shape)
        except KeyError as e:
            raise KeyError(
                "DaCapo expects Zarr datasets to have an 'offset' attribute! ",
                f"Zarr {self.file_name} and dataset {self.dataset} has attributes: {self._attributes}",
            ) from e

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
    def data(self):
        zarr_container = zarr.open(str(self.file_name))
        return zarr_container[self.dataset]

    def __getitem__(self, roi: Roi) -> np.ndarray:
        return daisy.open_ds(str(self.file_name), self.dataset, mode="r+").to_ndarray(
            roi=roi
        )

    def __setitem__(self, roi: Roi, value: np.ndarray):
        daisy.open_ds(str(self.file_name), self.dataset, mode="r+")[roi] = value

    @classmethod
    def create_from_array_identifier(
        cls,
        array_identifier,
        axes,
        roi,
        num_channels,
        voxel_size,
        dtype,
        chunks=None,
        name=None,
    ):
        """
        Create a new ZarrArray given an array identifier. It is assumed that
        this array_identifier points to a dataset that does not yet exist
        """
        zarr_container = zarr.open(array_identifier.container, "r+")
        zarr_dataset = zarr_container.create_dataset(
            array_identifier.dataset,
            shape=(num_channels,) + roi.shape / voxel_size
            if num_channels is not None
            else roi.shape / voxel_size,
            dtype=dtype,
            chunks=chunks,
        )
        zarr_dataset.attrs["offset"] = roi.offset
        zarr_dataset.attrs["resolution"] = voxel_size
        zarr_dataset.attrs["axes"] = axes

        zarr_array = cls.__new__(cls)
        zarr_array.file_name = array_identifier.container
        zarr_array.dataset = array_identifier.dataset
        zarr_array._attributes = zarr_array.data.attrs
        return zarr_array

    @classmethod
    def open_from_array_identifier(cls, array_identifier):
        zarr_array = cls.__new__(cls)
        zarr_array.file_name = array_identifier.container
        zarr_array.dataset = array_identifier.dataset
        zarr_array._attributes = zarr_array.data.attrs
        return zarr_array

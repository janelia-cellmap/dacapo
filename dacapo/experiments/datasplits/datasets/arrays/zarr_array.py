from .array import Array

from funlib.geometry import Coordinate, Roi
import daisy

import lazy_property
import numpy as np
import zarr

import logging

logger = logging.getLogger(__name__)


def voxel_size_conventions(attrs):
    if "resolution" in attrs:
        # Funkelab convention
        return Coordinate(attrs["resolution"])
    elif "pixelResolution" in attrs:
        # cosem convention
        return Coordinate(attrs["pixelResolution"]["dimensions"])
    elif "transform" in attrs:
        # also a cosem convention
        return Coordinate(attrs["transform"]["scale"])
    raise ValueError(
        "DaCapo expects the voxel size to be stored in the zarr metadata!\n"
        f"Attributes provided: {dict(attrs)}"
    )


def offset_conventions(attrs):
    if "offset" in attrs:
        # Funkelab convention
        return Coordinate(attrs["offset"])
    raise ValueError("DaCapo expects offset to tbe stored in the zarr metadata!")


class ZarrArray(Array):
    """This is a zarr array"""

    def __init__(self, array_config):
        super().__init__()
        self.file_name = array_config.file_name
        self.dataset = array_config.dataset

        self._attributes = self.data.attrs
        self._axes = array_config._axes

    @property
    def axes(self):
        if self._axes is not None:
            return self._axes
        try:
            return self._attributes["axes"]
        except KeyError as e:
            raise KeyError(
                "DaCapo expects Zarr datasets to have an 'axes' attribute!\n"
                f"Zarr {self.file_name} and dataset {self.dataset} has attributes: {self._attributes}",
            ) from e

    @property
    def dims(self) -> int:
        return self.voxel_size.dims

    @lazy_property.LazyProperty
    def voxel_size(self) -> Coordinate:
        try:
            return voxel_size_conventions(self._attributes)
        except KeyError as e:
            raise KeyError(
                "DaCapo expects Zarr datasets to have an 'resolution' attribute!\n"
                f"Zarr {self.file_name} and dataset {self.dataset} has attributes: {self._attributes}",
            ) from e

    @lazy_property.LazyProperty
    def roi(self) -> Roi:
        try:
            offset = offset_conventions(self._attributes)
        except ValueError:
            offset = Coordinate((0,) * self.dims)
            logger.warning(
                "Using a default of 0 offset since there is no offset stored in the zarr metadata!\n"
                f"Zarr {self.file_name} and dataset {self.dataset} has attributes: {dict(self._attributes)}"
            )
        shape = Coordinate(self.data.shape[-self.dims :]) * self.voxel_size
        return Roi(offset, shape)

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
        chunks=None,
        name=None,
    ):
        """
        Create a new ZarrArray given an array identifier. It is assumed that
        this array_identifier points to a dataset that does not yet exist
        """
        zarr_container = zarr.open(array_identifier.container, "a")
        try:
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
        return zarr_array

    @classmethod
    def open_from_array_identifier(cls, array_identifier):
        zarr_array = cls.__new__(cls)
        zarr_array.file_name = array_identifier.container
        zarr_array.dataset = array_identifier.dataset
        zarr_array._axes = None
        zarr_array._attributes = zarr_array.data.attrs
        return zarr_array

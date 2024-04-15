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
from upath import UPath as Path
import json
from typing import Dict, Tuple, Any, Optional, List

logger = logging.getLogger(__name__)


class ZarrArray(Array):
    """
    This is a zarr array.

    Attributes:
        name (str): The name of the array.
        file_name (Path): The file name of the array.
        dataset (str): The dataset name.
        _axes (Optional[List[str]]): The axes of the array.
        snap_to_grid (Optional[Coordinate]): The snap to grid.
    Methods:
        __init__(array_config):
            Initializes the array type 'raw' and name for the DummyDataset instance.
        __str__():
            Returns the string representation of the ZarrArray.
        __repr__():
            Returns the string representation of the ZarrArray.
        attrs():
            Returns the attributes of the array.
        axes():
            Returns the axes of the array.
        dims():
            Returns the dimensions of the array.
        _daisy_array():
            Returns the daisy array.
        voxel_size():
            Returns the voxel size of the array.
        roi():
            Returns the region of interest of the array.
        writable():
            Returns the boolean value of the array.
        dtype():
            Returns the data type of the array.
        num_channels():
            Returns the number of channels of the array.
        spatial_axes():
            Returns the spatial axes of the array.
        data():
            Returns the data of the array.
        __getitem__(roi):
            Returns the data of the array for the given region of interest.
        __setitem__(roi, value):
            Sets the data of the array for the given region of interest.
        create_from_array_identifier(array_identifier, axes, roi, num_channels, voxel_size, dtype, write_size=None, name=None, overwrite=False):
            Creates a new ZarrArray given an array identifier.
        open_from_array_identifier(array_identifier, name=""):
            Opens a new ZarrArray given an array identifier.
        _can_neuroglance():
            Returns the boolean value of the array.
        _neuroglancer_source():
            Returns the neuroglancer source of the array.
        _neuroglancer_layer():
            Returns the neuroglancer layer of the array.
        _transform_matrix():
            Returns the transform matrix of the array.
        _output_dimensions():
            Returns the output dimensions of the array.
        _source_name():
            Returns the source name of the array.
        add_metadata(metadata):
            Adds metadata to the array.
    Notes:
        This class is used to create a zarr array.
    """

    def __init__(self, array_config):
        """
        Initializes the array type 'raw' and name for the DummyDataset instance.

        Args:
            array_config (object): an instance of a configuration class that includes the name and
            raw configuration of the data.
        Raises:
            NotImplementedError
                If the method is not implemented in the derived class.
        Examples:
            >>> dataset = DummyDataset(dataset_config)
        Notes:
            This method is used to initialize the dataset.
        """
        super().__init__()
        self.name = array_config.name
        self.file_name = array_config.file_name
        self.dataset = array_config.dataset

        self._attributes = self.data.attrs
        self._axes = array_config._axes
        self.snap_to_grid = array_config.snap_to_grid

    def __str__(self):
        """
        Returns the string representation of the ZarrArray.

        Args:
            ZarrArray (str): The string representation of the ZarrArray.
        Returns:
            str: The string representation of the ZarrArray.
        Raises:
            NotImplementedError
        Examples:
            >>> print(ZarrArray)
        Notes:
            This method is used to return the string representation of the ZarrArray.
        """
        return f"ZarrArray({self.file_name}, {self.dataset})"

    def __repr__(self):
        """
        Returns the string representation of the ZarrArray.

        Args:
            ZarrArray (str): The string representation of the ZarrArray.
        Returns:
            str: The string representation of the ZarrArray.
        Raises:
            NotImplementedError
        Examples:
            >>> print(ZarrArray)
        Notes:
            This method is used to return the string representation of the ZarrArray.

        """
        return f"ZarrArray({self.file_name}, {self.dataset})"

    @property
    def attrs(self):
        """
        Returns the attributes of the array.

        Args:
            attrs (Any): The attributes of the array.
        Returns:
            Any: The attributes of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> attrs()
        Notes:
            This method is used to return the attributes of the array.

        """
        return self.data.attrs

    @property
    def axes(self):
        """
        Returns the axes of the array.

        Args:
            axes (List[str]): The axes of the array.
        Returns:
            List[str]: The axes of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> axes()
        Notes:
            This method is used to return the axes of the array.
        """
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
        """
        Returns the dimensions of the array.

        Args:
            dims (int): The dimensions of the array.
        Returns:
            int: The dimensions of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> dims()
        Notes:
            This method is used to return the dimensions of the array.

        """
        return self.voxel_size.dims

    @lazy_property.LazyProperty
    def _daisy_array(self) -> funlib.persistence.Array:
        """
        Returns the daisy array.

        Args:
            voxel_size (Coordinate): The voxel size.
        Returns:
            funlib.persistence.Array: The daisy array.
        Raises:
            NotImplementedError
        Examples:
            >>> _daisy_array()
        Notes:
            This method is used to return the daisy array.

        """
        return funlib.persistence.open_ds(f"{self.file_name}", self.dataset)

    @lazy_property.LazyProperty
    def voxel_size(self) -> Coordinate:
        """
        Returns the voxel size of the array.

        Args:
            voxel_size (Coordinate): The voxel size.
        Returns:
            Coordinate: The voxel size of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> voxel_size()
        Notes:
            This method is used to return the voxel size of the array.

        """
        return self._daisy_array.voxel_size

    @lazy_property.LazyProperty
    def roi(self) -> Roi:
        """
        Returns the region of interest of the array.

        Args:
            roi (Roi): The region of interest.
        Returns:
            Roi: The region of interest of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> roi()
        Notes:
            This method is used to return the region of interest of the array.
        """
        if self.snap_to_grid is not None:
            return self._daisy_array.roi.snap_to_grid(self.snap_to_grid, mode="shrink")
        else:
            return self._daisy_array.roi

    @property
    def writable(self) -> bool:
        """
        Returns the boolean value of the array.

        Args:
            writable (bool): The boolean value of the array.
        Returns:
            bool: The boolean value of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> writable()
        Notes:
            This method is used to return the boolean value of the array.
        """
        return True

    @property
    def dtype(self) -> Any:
        """
        Returns the data type of the array.

        Args:
            dtype (Any): The data type of the array.
        Returns:
            Any: The data type of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> dtype()
        Notes:
            This method is used to return the data type of the array.
        """
        return self.data.dtype

    @property
    def num_channels(self) -> Optional[int]:
        """
        Returns the number of channels of the array.

        Args:
            num_channels (Optional[int]): The number of channels of the array.
        Returns:
            Optional[int]: The number of channels of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> num_channels()
        Notes:
            This method is used to return the number of channels of the array.

        """
        return None if "c" not in self.axes else self.data.shape[self.axes.index("c")]

    @property
    def spatial_axes(self) -> List[str]:
        """
        Returns the spatial axes of the array.

        Args:
            spatial_axes (List[str]): The spatial axes of the array.
        Returns:
            List[str]: The spatial axes of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> spatial_axes()
        Notes:
            This method is used to return the spatial axes of the array.

        """
        return [ax for ax in self.axes if ax not in set(["c", "b"])]

    @property
    def data(self) -> Any:
        """
        Returns the data of the array.

        Args:
            data (Any): The data of the array.
        Returns:
            Any: The data of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> data()
        Notes:
            This method is used to return the data of the array.
        """
        zarr_container = zarr.open(str(self.file_name))
        return zarr_container[self.dataset]

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Returns the data of the array for the given region of interest.

        Args:
            roi (Roi): The region of interest.
        Returns:
            np.ndarray: The data of the array for the given region of interest.
        Raises:
            NotImplementedError
        Examples:
            >>> __getitem__(roi)
        Notes:
            This method is used to return the data of the array for the given region of interest.
        """
        data: np.ndarray = funlib.persistence.Array(
            self.data, self.roi, self.voxel_size
        ).to_ndarray(roi=roi)
        return data

    def __setitem__(self, roi: Roi, value: np.ndarray):
        """
        Sets the data of the array for the given region of interest.

        Args:
            roi (Roi): The region of interest.
            value (np.ndarray): The value to set.
        Raises:
            NotImplementedError
        Examples:
            >>> __setitem__(roi, value)
        Notes:
            This method is used to set the data of the array for the given region of interest.
        """
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
        this array_identifier points to a dataset that does not yet exist.

        Args:
            array_identifier (ArrayIdentifier): The array identifier.
            axes (List[str]): The axes of the array.
            roi (Roi): The region of interest.
            num_channels (int): The number of channels.
            voxel_size (Coordinate): The voxel size.
            dtype (Any): The data type.
            write_size (Optional[Coordinate]): The write size.
            name (Optional[str]): The name of the array.
            overwrite (bool): The boolean value to overwrite the array.
        Returns:
            ZarrArray: The ZarrArray.
        Raises:
            NotImplementedError
        Examples:
            >>> create_from_array_identifier(array_identifier, axes, roi, num_channels, voxel_size, dtype, write_size=None, name=None, overwrite=False)
        Notes:
            This method is used to create a new ZarrArray given an array identifier.
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
        """
        Opens a new ZarrArray given an array identifier.

        Args:
            array_identifier (ArrayIdentifier): The array identifier.
            name (str): The name of the array.
        Returns:
            ZarrArray: The ZarrArray.
        Raises:
            NotImplementedError
        Examples:
            >>> open_from_array_identifier(array_identifier, name="")
        Notes:
            This method is used to open a new ZarrArray given an array identifier.
        """
        zarr_array = cls.__new__(cls)
        zarr_array.name = name
        zarr_array.file_name = array_identifier.container
        zarr_array.dataset = array_identifier.dataset
        zarr_array._axes = None
        zarr_array._attributes = zarr_array.data.attrs
        zarr_array.snap_to_grid = None
        return zarr_array

    def _can_neuroglance(self) -> bool:
        """
        Returns the boolean value of the array.

        Args:
            can_neuroglance (bool): The boolean value of the array.
        Returns:
            bool: The boolean value of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> can_neuroglance()
        Notes:
            This method is used to return the boolean value of the array.
        """
        return True

    def _neuroglancer_source(self):
        """
        Returns the neuroglancer source of the array.

        Args:
            neuroglancer.LocalVolume: The neuroglancer source of the array.
        Returns:
            neuroglancer.LocalVolume: The neuroglancer source of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> neuroglancer_source()
        Notes:
            This method is used to return the neuroglancer source of the array.

        """
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
        """
        Returns the neuroglancer layer of the array.

        Args:
            layer (neuroglancer.ImageLayer): The neuroglancer layer of the array.
        Returns:
            Tuple[neuroglancer.ImageLayer, Dict[str, Any]]: The neuroglancer layer of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> neuroglancer_layer()
        Notes:
            This method is used to return the neuroglancer layer of the array.
        """
        layer = neuroglancer.ImageLayer(source=self._neuroglancer_source())
        return layer

    def _transform_matrix(self):
        """
        Returns the transform matrix of the array.

        Args:
            transform_matrix (List[List[float]]): The transform matrix of the array.
        Returns:
            List[List[float]]: The transform matrix of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> transform_matrix()
        Notes:
            This method is used to return the transform matrix of the array.
        """
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
        """
        Returns the output dimensions of the array.

        Args:
            output_dimensions (Dict[str, Tuple[float, str]]): The output dimensions of the array.
        Returns:
            Dict[str, Tuple[float, str]]: The output dimensions of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> output_dimensions()
        Notes:
            This method is used to return the output dimensions of the array.
        """
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
        """
        Returns the source name of the array.

        Args:
            source_name (str): The source name of the array.
        Returns:
            str: The source name of the array.
        Raises:
            NotImplementedError
        Examples:
            >>> source_name()
        Notes:
            This method is used to return the source name of the array.

        """
        return self.name

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Adds metadata to the array.

        Args:
            metadata (Dict[str, Any]): The metadata to add to the array.
        Raises:
            NotImplementedError
        Examples:
            >>> add_metadata(metadata)
        Notes:
            This method is used to add metadata to the array.

        """
        dataset = zarr.open(self.file_name, mode="a")[self.dataset]
        for k, v in metadata.items():
            dataset.attrs[k] = v

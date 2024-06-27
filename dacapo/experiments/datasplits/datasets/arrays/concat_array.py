from .array import Array

from funlib.geometry import Roi

import numpy as np

from typing import Dict, Any
import logging

logger = logging.getLogger(__file__)


class ConcatArray(Array):
    """
    This is a wrapper around other `source_arrays` that concatenates
    them along the channel dimension. The `source_arrays` are expected
    to have the same shape and ROI, but can have different data types.

    Attributes:
        name: The name of the array.
        channels: The list of channel names.
        source_arrays: A dictionary mapping channel names to source arrays.
        default_array: An optional default array to use for channels that are
            not present in `source_arrays`.
    Methods:
        from_toml(cls, toml_path: str) -> ConcatArrayConfig:
            Load the ConcatArrayConfig from a TOML file
        to_toml(self, toml_path: str) -> None:
            Save the ConcatArrayConfig to a TOML file
        create_array(self) -> ConcatArray:
            Create the ConcatArray from the config
    Note:
        This class is a subclass of Array and inherits all its attributes
        and methods. The only difference is that the array_type is ConcatArray.

    """

    def __init__(self, array_config):
        """
        Initialize the ConcatArray from a ConcatArrayConfig.

        Args:
            array_config (ConcatArrayConfig): The config to create the ConcatArray from.
        Raises:
            AssertionError: If the source arrays have different shapes or ROIs.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
        Note:
            The `source_arrays` are expected to have the same shape and ROI,
            but can have different data types.
        """
        self.name = array_config.name
        self.channels = array_config.channels
        self.source_arrays = {
            channel: source_array_config.array_type(source_array_config)
            for channel, source_array_config in array_config.source_array_configs.items()
        }
        self.default_array = (
            array_config.default_config.array_type(array_config.default_config)
            if array_config.default_config is not None
            else None
        )

    @property
    def attrs(self):
        """
        Return the attributes of the ConcatArray as a dictionary.

        Returns:
            Dict[str, Any]: The attributes of the ConcatArray.
        Raises:
            AssertionError: If the source arrays have different attributes.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.attrs
            {'axes': 'cxyz', 'roi': Roi(...), 'voxel_size': (1, 1, 1)}
        Note:
            The `source_arrays` are expected to have the same attributes.
        """
        return dict()

    @property
    def source_arrays(self) -> Dict[str, Array]:
        """
        Return the source arrays of the ConcatArray.

        Returns:
            Dict[str, Array]: The source arrays of the ConcatArray.
        Raises:
            AssertionError: If the source arrays are empty.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.source_arrays
            {'A': Array(...), 'B': Array(...)}
        Note:
            The `source_arrays` are expected to have the same shape and ROI.
        """
        return self._source_arrays

    @source_arrays.setter
    def source_arrays(self, value: Dict[str, Array]):
        """
        Set the source arrays of the ConcatArray.

        Args:
            value (Dict[str, Array]): The source arrays to set.
        Raises:
            AssertionError: If the source arrays are empty.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.source_arrays = {'A': Array(...), 'B': Array(...)}
        Note:
            The `source_arrays` are expected to have the same shape and ROI.
        """
        assert len(value) > 0, "Source arrays is empty!"
        self._source_arrays = value
        attrs: Dict[str, Any] = {}
        for source_array in value.values():
            axes = attrs.get("axes", source_array.axes)
            assert source_array.axes == axes
            assert axes[0] == "c" or "c" not in axes
            attrs["axes"] = axes
            roi = attrs.get("roi", source_array.roi)
            assert not (not roi.empty and source_array.roi.intersect(roi).empty), (
                self.name,
                [x.roi for x in self._source_arrays.values()],
            )
            attrs["roi"] = source_array.roi.intersect(roi)
            voxel_size = attrs.get("voxel_size", source_array.voxel_size)
            assert source_array.voxel_size == voxel_size
            attrs["voxel_size"] = voxel_size
        self._source_array = source_array

    @property
    def source_array(self) -> Array:
        """
        Return the source array of the ConcatArray.

        Returns:
            Array: The source array of the ConcatArray.
        Raises:
            AssertionError: If the source array is None.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.source_array
            Array(...)
        Note:
            The `source_array` is expected to have the same shape and ROI.
        """
        return self._source_array

    @property
    def axes(self):
        """
        Return the axes of the ConcatArray.

        Returns:
            str: The axes of the ConcatArray.
        Raises:
            AssertionError: If the source arrays have different axes.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.axes
            'cxyz'
        Note:
            The `source_arrays` are expected to have the same axes.
        """
        source_axes = self.source_array.axes
        if "c" not in source_axes:
            source_axes = ["c"] + source_axes
        return source_axes

    @property
    def dims(self):
        """
        Return the dimensions of the ConcatArray.

        Returns:
            Tuple[int]: The dimensions of the ConcatArray.
        Raises:
            AssertionError: If the source arrays have different dimensions.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.dims
            (2, 100, 100, 100)
        Note:
            The `source_arrays` are expected to have the same dimensions.
        """
        return self.source_array.dims

    @property
    def voxel_size(self):
        """
        Return the voxel size of the ConcatArray.

        Returns:
            Tuple[float]: The voxel size of the ConcatArray.
        Raises:
            AssertionError: If the source arrays have different voxel sizes.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.voxel_size
            (1, 1, 1)
        Note:
            The `source_arrays` are expected to have the same voxel size.
        """
        return self.source_array.voxel_size

    @property
    def roi(self):
        """
        Return the ROI of the ConcatArray.

        Returns:
            Roi: The ROI of the ConcatArray.
        Raises:
            AssertionError: If the source arrays have different ROIs.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.roi
            Roi(...)
        Note:
            The `source_arrays` are expected to have the same ROI.
        """
        return self.source_array.roi

    @property
    def writable(self) -> bool:
        """
        Return whether the ConcatArray is writable.

        Returns:
            bool: Whether the ConcatArray is writable.
        Raises:
            AssertionError: If the ConcatArray is writable.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.writable
            False
        Note:
            The ConcatArray is not writable.
        """
        return False

    @property
    def data(self):
        """
        Return the data of the ConcatArray.

        Returns:
            np.ndarray: The data of the ConcatArray.
        Raises:
            RuntimeError: If the ConcatArray is not writable.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.data
            np.ndarray(...)
        Note:
            The ConcatArray is not writable.
        """
        raise RuntimeError("Cannot get writable version of this data!")

    @property
    def dtype(self):
        """
        Return the data type of the ConcatArray.

        Returns:
            np.dtype: The data type of the ConcatArray.
        Raises:
            AssertionError: If the source arrays have different data types.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.dtype
            np.float32
        Note:
            The `source_arrays` are expected to have the same data type.
        """
        return self.source_array.dtype

    @property
    def num_channels(self):
        """
        Return the number of channels of the ConcatArray.

        Returns:
            int: The number of channels of the ConcatArray.
        Raises:
            AssertionError: If the source arrays have different numbers of channels.
        Examples:
            >>> config = ConcatArrayConfig(
            ...     name="my_concat_array",
            ...     channels=["A", "B"],
            ...     source_array_configs={
            ...         "A": ArrayConfig(...),
            ...         "B": ArrayConfig(...),
            ...     },
            ...     default_config=ArrayConfig(...),
            ... )
            >>> array = ConcatArray(config)
            >>> array.num_channels
            2
        Note:
            The `source_arrays` are expected to have the same number of channels.
        """
        return len(self.channels)

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Return the data of the ConcatArray for a given ROI.

        Args:
            roi (Roi): The ROI to get the data for.
        Returns:
            np.ndarray: The data of the ConcatArray for the given ROI.
        Raises:
            AssertionError: If the source arrays have different shapes or ROIs.
        Examples:
            >>> roi = Roi(...)
            >>> array[roi]
            np.ndarray(...)
        Note:
            The `source_arrays` are expected to have the same shape and ROI.
        """
        default = (
            np.zeros_like(self.source_array[roi])
            if self.default_array is None
            else self.default_array[roi]
        )
        arrays = [
            (
                self.source_arrays[channel][roi]
                if channel in self.source_arrays
                else default
            )
            for channel in self.channels
        ]
        shapes = [array.shape for array in arrays]
        ndims = max([len(shape) for shape in shapes])
        assert ndims <= len(self.axes), f"{self.axes}, {ndims}"
        shapes = [(1,) * (len(self.axes) - len(shape)) + shape for shape in shapes]
        for axis_shapes in zip(*shapes):
            assert max(axis_shapes) == min(axis_shapes), f"{shapes}"
        arrays = [array.reshape(shapes[0]) for array in arrays]
        concatenated = np.concatenate(
            arrays,
            axis=0,
        )
        if concatenated.shape[0] == 1:
            print(
                f"Concatenated array has only one channel: {self.name} {concatenated.shape}"
            )
        return concatenated

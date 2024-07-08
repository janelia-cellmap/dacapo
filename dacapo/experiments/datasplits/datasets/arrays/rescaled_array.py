from .array import Array

import funlib.persistence
from funlib.geometry import Coordinate, Roi

import numpy as np
from skimage.transform import rescale


class RescaledArray(Array):
    """
    This is a zarr array that is a rescaled version of another array.

    Rescaling is done by rescaling the source array to the given
    voxel size.

    Attributes:
        name: str
            The name of the array
        source_array: Array
            The source array
        target_voxel_size: tuple[int | float]
            The target voxel size
        interp_order: int
            The order of the interpolation used for resampling
    Methods:
        attrs: Dict
            Returns the attributes of the source array
        axes: str
            Returns the axes of the source array
        dims: int
            Returns the number of dimensions of the source array
        voxel_size: Coordinate
            Returns the voxel size of the rescaled array
        roi: Roi
            Returns the region of interest of the rescaled array
        writable: bool
            Returns whether the rescaled array is writable
        dtype: np.dtype
            Returns the data type of the rescaled array
        num_channels: int
            Returns the number of channels of the rescaled array
        data: np.ndarray
            Returns the data of the rescaled array
        scale: Tuple[float]
            Returns the scale of the rescaled array
        __getitem__(roi: Roi) -> np.ndarray
            Returns the data of the rescaled array within the given region of interest
        _can_neuroglance() -> bool
            Returns whether the source array can be visualized with neuroglance
        _neuroglancer_layer() -> Dict
            Returns the neuroglancer layer of the source array
        _neuroglancer_source() -> Dict
            Returns the neuroglancer source of the source array
        _source_name() -> str
            Returns the name of the source array
    Note:
        This class is a subclass of Array.


    """

    def __init__(self, array_config):
        """
        Constructor of the RescaledArray class.

        Args:
            array_config: ArrayConfig
                The configuration of the array
        Examples:
            >>> rescaled_array = RescaledArray(array_config)
        Note:
            This constructor resamples the source array to the given voxel size.
        """
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

        self.target_voxel_size = array_config.target_voxel_size
        self.interp_order = array_config.interp_order

    @property
    def attrs(self):
        """
        Returns the attributes of the source array.

        Returns:
            Dict: The attributes of the source array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array.attrs
        Note:
            This method returns the attributes of the source array.

        """
        return self._source_array.attrs

    @property
    def axes(self):
        """
        Returns the axes of the source array.

        Returns:
            str: The axes of the source array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array.axes
        Note:
            This method returns the axes of the source array.
        """
        return self._source_array.axes

    @property
    def dims(self) -> int:
        """
        Returns the number of dimensions of the source array.

        Returns:
            int: The number of dimensions of the source array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array.dims
        Note:
            This method returns the number of dimensions of the source array.
        """
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        """
        Returns the voxel size of the rescaled array.

        Returns:
            Coordinate: The voxel size of the rescaled array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array.voxel_size
        Note:
            This method returns the voxel size of the rescaled array.
        """
        return self.target_voxel_size

    @property
    def roi(self) -> Roi:
        """
        Returns the region of interest of the rescaled array.

        Returns:
            Roi: The region of interest of the rescaled array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array.roi
        Note:
            This method returns the region of interest of the rescaled array.

        """
        return self._source_array.roi.snap_to_grid(self.voxel_size, mode="shrink")

    @property
    def writable(self) -> bool:
        """
        Returns whether the rescaled array is writable.

        Returns:
            bool: True if the rescaled array is writable, False otherwise
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array.writable
        Note:
            This method returns whether the rescaled array is writable.

        """
        return False

    @property
    def dtype(self):
        """
        Returns the data type of the rescaled array.

        Returns:
            np.dtype: The data type of the rescaled array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array.dtype
        Note:
            This method returns the data type of the rescaled array.
        """
        return self._source_array.dtype

    @property
    def num_channels(self) -> int:
        """
        Returns the number of channels of the rescaled array.

        Returns:
            int: The number of channels of the rescaled array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array.num_channels
        Note:
            This method returns the number of channels of the rescaled array.
        """
        return self._source_array.num_channels

    @property
    def data(self):
        """
        Returns the data of the rescaled array.

        Returns:
            np.ndarray: The data of the rescaled array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array.data
        Note:
            This method returns the data of the rescaled array.
        """
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    @property
    def scale(self):
        """
        Returns the scaling factor of the rescaled array.

        Returns:
            Tuple[float]: The scaling factor of the rescaled array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array.scale
        Note:
            This method returns the scaling factor of the rescaled array.

        """
        spatial_scales = tuple(
            o / r for r, o in zip(self.target_voxel_size, self._source_array.voxel_size)
        )
        if "c" in self.axes:
            scales = list(spatial_scales)
            scales.insert(self.axes.index("c"), 1.0)
            return tuple(scales)
        else:
            return spatial_scales

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Returns the data of the rescaled array within the given region of interest.

        Args:
            roi: Roi
                The region of interest
        Returns:
            np.ndarray: The data of the rescaled array within the given region of interest
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array[roi]
        Note:
            This method returns the data of the rescaled array within the given region of interest.
        """
        snapped_roi = roi.snap_to_grid(self._source_array.voxel_size, mode="grow")
        rescaled_array = funlib.persistence.Array(
            rescale(
                self._source_array[snapped_roi].astype(np.float32),
                self.scale,
                order=self.interp_order,
                anti_aliasing=self.interp_order != 0,
            ).astype(self.dtype),
            roi=snapped_roi,
            voxel_size=self.voxel_size,
        )
        return rescaled_array.to_ndarray(roi)

    def _can_neuroglance(self):
        """
        Returns whether the source array can be visualized with neuroglance.

        Returns:
            bool: True if the source array can be visualized with neuroglance, False otherwise
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array._can_neuroglance()
        Note:
            This method returns whether the source array can be visualized with neuroglance.
        """
        return self._source_array._can_neuroglance()

    def _neuroglancer_layer(self):
        """
        Returns the neuroglancer layer of the source array.

        Returns:
            Dict: The neuroglancer layer of the source array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array._neuroglancer_layer()
        Note:
            This method returns the neuroglancer layer of the source array.
        """
        return self._source_array._neuroglancer_layer()

    def _neuroglancer_source(self):
        """
        Returns the neuroglancer source of the source array.

        Returns:
            Dict: The neuroglancer source of the source array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array._neuroglancer_source()
        Note:
            This method returns the neuroglancer source of the source array.
        """
        return self._source_array._neuroglancer_source()

    def _source_name(self):
        """
        Returns the name of the source array.

        Returns:
            str: The name of the source array
        Raises:
            ValueError: If the rescaled array is not writable
        Examples:
            >>> rescaled_array._source_name()
        Note:
            This method returns the name of the source array.
        """
        return self._source_array._source_name()

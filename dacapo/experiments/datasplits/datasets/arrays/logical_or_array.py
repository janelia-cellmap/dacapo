from .array import Array

from funlib.geometry import Coordinate, Roi


import neuroglancer

import numpy as np


class LogicalOrArray(Array):
    """
    Array that computes the logical OR of the instances in a list of source arrays.

    Attributes:
        name: str
            The name of the array
        source_array: Array
            The source array from which to take the logical OR
    Methods:
        axes: () -> List[str]
            Get the axes of the array
        dims: () -> int
            Get the number of dimensions of the array
        voxel_size: () -> Coordinate
            Get the voxel size of the array
        roi: () -> Roi
            Get the region of interest of the array
        writable: () -> bool
            Get whether the array is writable
        dtype: () -> type
            Get the data type of the array
        num_channels: () -> int
            Get the number of channels in the array
        data: () -> np.ndarray
            Get the data of the array
        attrs: () -> dict
            Get the attributes of the array
        __getitem__: (roi: Roi) -> np.ndarray
            Get the data of the array in the region of interest
        _can_neuroglance: () -> bool
            Get whether the array can be visualized in neuroglance
        _neuroglancer_source: () -> dict
            Get the neuroglancer source of the array
        _neuroglancer_layer: () -> Tuple[neuroglancer.Layer, dict]
            Get the neuroglancer layer of the array
        _source_name: () -> str
            Get the name of the source array
    Notes:
        The LogicalOrArray class is used to create a LogicalOrArray. The LogicalOrArray
        class is a subclass of the Array class.
    """

    def __init__(self, array_config):
        """
        Create a LogicalOrArray instance from a configuration
        Args:
            array_config: MergeInstancesArrayConfig
                The configuration for the array
        Returns:
            LogicalOrArray
                The LogicalOrArray instance created from the configuration
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array.name
            'logical_or'
            >>> array.source_array.name
            'mask1'
            >>> array.source_array.mask_id
            1
        Notes:
            The create_array method is used to create a LogicalOrArray instance from a
            configuration. The LogicalOrArray instance is created by taking the logical OR
            of the instances in the source arrays.
        """
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

    @property
    def axes(self):
        """
        Get the axes of the array

        Returns:
            List[str]: The axes of the array
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array.axes
            ['x', 'y', 'z']
        Notes:
            The axes method is used to get the axes of the array. The axes are the dimensions
            of the array.
        """
        return [x for x in self._source_array.axes if x != "c"]

    @property
    def dims(self) -> int:
        """
        Get the number of dimensions of the array

        Returns:
            int: The number of dimensions of the array
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array.dims
            3
        Notes:
            The dims method is used to get the number of dimensions of the array. The number
            of dimensions is the number of axes of the array.
        """
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        """
        Get the voxel size of the array

        Returns:
            Coordinate: The voxel size of the array
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array.voxel_size
            Coordinate(x=1.0, y=1.0, z=1.0)
        Notes:
            The voxel_size method is used to get the voxel size of the array. The voxel size
            is the size of a voxel in the array.

        """
        return self._source_array.voxel_size

    @property
    def roi(self) -> Roi:
        """
        Get the region of interest of the array

        Returns:
            Roi: The region of interest of the array
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array.roi
            Roi(offset=(0, 0, 0), shape=(10, 10, 10))
        Notes:
            The roi method is used to get the region of interest of the array. The region of
            interest is the shape and offset of the array.
        """
        return self._source_array.roi

    @property
    def writable(self) -> bool:
        """
        Get whether the array is writable

        Returns:
            bool: Whether the array is writable
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array.writable
            False
        Notes:
            The writable method is used to get whether the array is writable. An array is
            writable if it can be modified.
        """
        return False

    @property
    def dtype(self):
        """
        Get the data type of the array

        Returns:
            type: The data type of the array
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array.dtype
            <class 'numpy.uint8'>
        Notes:
            The dtype method is used to get the data type of the array. The data type is the
            type of the data in the array.
        """
        return np.uint8

    @property
    def num_channels(self):
        """
        Get the number of channels in the array

        Returns:
            int: The number of channels in the array
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array.num_channels
            1
        Notes:
            The num_channels method is used to get the number of channels in the array. The
            number of channels is the number of channels in the array.
        """
        return None

    @property
    def data(self):
        """
        Get the data of the array

        Returns:
            np.ndarray: The data of the array
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array.data
            array([[[1, 1, 1, ..., 1, 1, 1],
                    [1, 1, 1, ..., 1, 1, 1],
                    [1, 1, 1, ..., 1, 1, 1],
                    ...,
                    [1, 1, 1, ..., 1, 1, 1],
                    [1, 1, 1, ..., 1, 1, 1],
                    [1, 1, 1, ..., 1, 1, 1]]], dtype=uint8)
        Notes:
            The data method is used to get the data of the array. The data is the content of
            the array.

        """
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    @property
    def attrs(self):
        """
        Get the attributes of the array

        Returns:
            dict: The attributes of the array
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array.attrs
            {'name': 'logical_or'}
        Notes:
            The attrs method is used to get the attributes of the array. The attributes are
            the metadata of the array.
        """
        return self._source_array.attrs

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Get the data of the array in the region of interest

        Args:
            roi: Roi
                The region of interest of the array
        Returns:
            np.ndarray: The data of the array in the region of interest
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> roi = Roi((0, 0, 0), (10, 10, 10))
            >>> array[roi]
            array([[[1, 1, 1, ..., 1, 1, 1],
                    [1, 1, 1, ..., 1, 1, 1],
                    [1, 1, 1, ..., 1, 1, 1],
                    ...,
                    [1, 1, 1, ..., 1, 1, 1],
                    [1, 1, 1, ..., 1, 1, 1],
                    [1, 1, 1, ..., 1, 1, 1]]], dtype=uint8)
        Notes:
            The __getitem__ method is used to get the data of the array in the region of interest.
            The data is the content of the array in the region of interest.
        """
        mask = self._source_array[roi]
        if "c" in self._source_array.axes:
            mask = np.max(mask, axis=self._source_array.axes.index("c"))
        return mask

    def _can_neuroglance(self):
        """
        Get whether the array can be visualized in neuroglance

        Returns:
            bool: Whether the array can be visualized in neuroglance
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array._can_neuroglance()
            True
        Notes:
            The _can_neuroglance method is used to get whether the array can be visualized
            in neuroglance.
        """
        return self._source_array._can_neuroglance()

    def _neuroglancer_source(self):
        """
        Get the neuroglancer source of the array

        Returns:
            dict: The neuroglancer source of the array
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array._neuroglancer_source()
            {'source': 'precomputed://https://mybucket.storage.googleapis.com/path/to/logical_or'}
        Notes:
            The _neuroglancer_source method is used to get the neuroglancer source of the array.
            The neuroglancer source is the source that is displayed in the neuroglancer viewer.
        """
        # source_arrays
        if hassattr(self._source_array, "source_arrays"):
            source_arrays = list(self._source_array.source_arrays)
            # apply logical or
            mask = np.logical_or.reduce(source_arrays)
            return mask
        return self._source_array._neuroglancer_source()

    def _combined_neuroglancer_source(self) -> neuroglancer.LocalVolume:
        """
        Combines dimensions and metadata from self._source_array._neuroglancer_source()
        with data from self._neuroglancer_source().

        Returns:
            neuroglancer.LocalVolume: The combined neuroglancer source.
        """
        source_array_volume = self._source_array._neuroglancer_source()
        if isinstance(source_array_volume,list):
            source_array_volume = source_array_volume[0]
        result_data = self._neuroglancer_source()
        
        return neuroglancer.LocalVolume(
            data=result_data,
            dimensions=source_array_volume.dimensions,
            voxel_offset=source_array_volume.voxel_offset,
        )

    def _neuroglancer_layer(self):
        """
        This method returns the neuroglancer layer for the source array.

        Returns:
            neuroglancer.SegmentationLayer: The neuroglancer layer for the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array._neuroglancer_layer()
        Note:
            This method is used to return the neuroglancer layer for the source array.
        """
        # layer = neuroglancer.SegmentationLayer(source=self._neuroglancer_source())
        return neuroglancer.SegmentationLayer(source=self._combined_neuroglancer_source())

    def _source_name(self):
        """
        Get the name of the source array

        Returns:
            str: The name of the source array
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> array_config = MergeInstancesArrayConfig(
            ...     name="logical_or",
            ...     source_array_configs=[
            ...         ArrayConfig(
            ...             name="mask1",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask1",
            ...                 mask_id=1,
            ...             ),
            ...         ),
            ...         ArrayConfig(
            ...             name="mask2",
            ...             array_type=MaskArray,
            ...             source_array_config=MaskArrayConfig(
            ...                 name="mask2",
            ...                 mask_id=2,
            ...             ),
            ...         ),
            ...     ],
            ... )
            >>> array = array_config.create_array()
            >>> array._source_name()
            'mask1'
        Notes:
            The _source_name method is used to get the name of the source array. The name
            of the source array is the name of the array that is being modified.
        """
        name = self._source_array._source_name()
        if isinstance(name, list):
            name = "_".join(name)
        return "logical_or"+name

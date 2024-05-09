from .array import Array

from funlib.geometry import Coordinate, Roi


import neuroglancer

import numpy as np


class MergeInstancesArray(Array):
    """
    This array merges multiple source arrays into a single array by summing them. This is useful for merging
    instance segmentation arrays into a single array. NeuoGlancer will display each instance as a different color.

    Attributes:
        name : str
            The name of the array
        source_array_configs : List[ArrayConfig]
            A list of source arrays to merge
    Methods:
        __getitem__(roi: Roi) -> np.ndarray
            Returns a numpy array with the requested region of interest
        _can_neuroglance() -> bool
            Returns True if the array can be visualized in neuroglancer
        _neuroglancer_source() -> str
            Returns the source name for the array in neuroglancer
        _neuroglancer_layer() -> Tuple[neuroglancer.SegmentationLayer, Dict[str, Any]]
            Returns a neuroglancer layer and its configuration
        _source_name() -> str
            Returns the source name for the array
    Note:
        This array is not writable
        Source arrays must have the same shape.

    """

    def __init__(self, array_config):
        """
        Constructor for MergeInstancesArray

        Args:
            array_config : MergeInstancesArrayConfig
                The configuration for the array
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            ```
        Note:
            This example shows how to create a MergeInstancesArray object
        """
        self.name = array_config.name
        self._source_arrays = [
            source_config.array_type(source_config)
            for source_config in array_config.source_array_configs
        ]
        self._source_array = self._source_arrays[0]

    @property
    def axes(self):
        """
        Returns the axes of the array

        Returns:
            List[str]: The axes of the array
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            axes = array.axes
            ```
        Note:
            This example shows how to get the axes of the array

        """
        return [x for x in self._source_array.axes if x != "c"]

    @property
    def dims(self) -> int:
        """
        Returns the number of dimensions of the array

        Returns:
            int: The number of dimensions of the array
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            dims = array.dims
            ```
        Note:
            This example shows how to get the number of dimensions of the array


        """
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        """
        Returns the voxel size of the array

        Returns:
            Coordinate: The voxel size of the array
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            voxel_size = array.voxel_size
            ```
        Note:
            This example shows how to get the voxel size of the array
        """
        return self._source_array.voxel_size

    @property
    def roi(self) -> Roi:
        """
        Returns the region of interest of the array

        Returns:
            Roi: The region of interest of the array
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            roi = array.roi
            ```
        Note:
            This example shows how to get the region of interest of the array
        """
        return self._source_array.roi

    @property
    def writable(self) -> bool:
        """
        Returns True if the array is writable, False otherwise

        Returns:
            bool: True if the array is writable, False otherwise
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            writable = array.writable
            ```
        Note:
            This example shows how to check if the array is writable
        """
        return False

    @property
    def dtype(self):
        """
        Returns the data type of the array

        Returns:
            np.dtype: The data type of the array
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            dtype = array.dtype
            ```
        Note:
            This example shows how to get the data type of the array
        """
        return np.uint8

    @property
    def num_channels(self):
        """
        Returns the number of channels of the array

        Returns:
            int: The number of channels of the array
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            num_channels = array.num_channels
            ```
        Note:
            This example shows how to get the number of channels of the array
        """
        return None

    @property
    def data(self):
        """
        Returns the data of the array

        Returns:
            np.ndarray: The data of the array
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            data = array.data
            ```
        Note:
            This example shows how to get the data of the array
        """
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    @property
    def attrs(self):
        """
        Returns the attributes of the array

        Returns:
            Dict[str, Any]: The attributes of the array
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            attributes = array.attrs
            ```
        Note:
            This example shows how to get the attributes of the array
        """
        return self._source_array.attrs

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Returns a numpy array with the requested region of interest

        Args:
            roi : Roi
                The region of interest to get
        Returns:
            np.ndarray: A numpy array with the requested region of interest
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            roi = Roi((0, 0, 0), (100, 100, 100))
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            array_data = array[roi]
            ```
        Note:
            This example shows how to get a numpy array with the requested region of interest
        """
        arrays = [source_array[roi] for source_array in self._source_arrays]
        offset = 0
        for array in arrays:
            array[array > 0] += offset
            offset = array.max()
        return np.sum(arrays, axis=0)

    def _can_neuroglance(self):
        """
        Returns True if the array can be visualized in neuroglancer, False otherwise

        Returns:
            bool: True if the array can be visualized in neuroglancer, False otherwise
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            can_neuroglance = array._can_neuroglance()
            ```
        Note:
            This example shows how to check if the array can be visualized in neuroglancer
        """
        return self._source_array._can_neuroglance()

    def _neuroglancer_source(self):
        """
        Returns the source name for the array in neuroglancer

        Returns:
            str: The source name for the array in neuroglancer
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            source = array._neuroglancer_source()
            ```
        Note:
            This example shows how to get the source name for the array in neuroglancer
        """
        return self._source_array._neuroglancer_source()

    def _neuroglancer_layer(self):
        """
        Returns a neuroglancer layer and its configuration

        Returns:
            Tuple[neuroglancer.SegmentationLayer, Dict[str, Any]]: A neuroglancer layer and its configuration
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            layer, kwargs = array._neuroglancer_layer()
            ```
        Note:
            This example shows how to get a neuroglancer layer and its configuration
        """
        # Generates an Segmentation layer

        layer = neuroglancer.SegmentationLayer(source=self._neuroglancer_source())
        kwargs = {
            "visible": False,
        }
        return layer, kwargs

    def _source_name(self):
        """
        Returns the source name for the array

        Returns:
            str: The source name for the array
        Raises:
            ValueError: If the source arrays have different shapes
        Example:
            ```python
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArray
            from dacapo.experiments.datasplits.datasets.arrays import MergeInstancesArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            from dacapo.experiments.datasplits.datasets.arrays import ArrayType
            from funlib.geometry import Coordinate, Roi
            array_config = MergeInstancesArrayConfig(
                name="array",
                source_array_configs=[
                    ArrayConfig(
                        name="array1",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array1.h5",
                    ),
                    ArrayConfig(
                        name="array2",
                        array_type=ArrayType.INSTANCE_SEGMENTATION,
                        path="path/to/array2.h5",
                    ),
                ],
            )
            array = MergeInstancesArray(array_config)
            source_name = array._source_name()
            ```
        Note:
            This example shows how to get the source name for the array
        """
        return self._source_array._source_name()

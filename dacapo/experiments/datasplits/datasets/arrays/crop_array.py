from .array import Array

from funlib.geometry import Coordinate, Roi

import numpy as np


class CropArray(Array):
    """
    Used to crop a larger array to a smaller array. This is useful when you
    want to work with a subset of a larger array, but don't want to copy the
    data. The crop is done on demand, so the data is not copied until you
    actually access it.

    Attributes:
        name: The name of the array.
        source_array: The array to crop.
        crop_roi: The region of interest to crop to.
    Methods:
        attrs: Returns the attributes of the source array.
        axes: Returns the axes of the source array.
        dims: Returns the number of dimensions of the source array.
        voxel_size: Returns the voxel size of the source array.
        roi: Returns the region of interest of the source array.
        writable: Returns whether the array is writable.
        dtype: Returns the data type of the source array.
        num_channels: Returns the number of channels of the source array.
        data: Returns the data of the source array.
        channels: Returns the channels of the source array.
        __getitem__(roi): Returns the data of the source array within the
            region of interest.
        _can_neuroglance(): Returns whether the source array can be viewed in
            Neuroglancer.
        _neuroglancer_source(): Returns the source of the source array for
            Neuroglancer.
        _neuroglancer_layer(): Returns the layer of the source array for
            Neuroglancer.
        _source_name(): Returns the name of the source array.
    Note:
        This class is a subclass of Array.


    """

    def __init__(self, array_config):
        """
        Initializes the CropArray.

        Args:
            array_config: The configuration of the array to crop.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
        Note:
            The source array configuration must be an instance of ArrayConfig.
        """
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )
        self.crop_roi = array_config.roi

    @property
    def attrs(self):
        """
        Returns the attributes of the source array.

        Returns:
            The attributes of the source array.
        Raises:
            ValueError: If the region of interest to crop to is not within the
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array.attrs
            {}
        Note:
            The attributes are empty because the source array is not modified.
        """
        return self._source_array.attrs

    @property
    def axes(self):
        """
        Returns the axes of the source array.

        Returns:
            The axes of the source array.
        Raises:
            ValueError: If the region of interest to crop to is not within the
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array.axes
            'zyx'
        Note:
            The axes are 'zyx' because the source array is not modified.
        """
        return self._source_array.axes

    @property
    def dims(self) -> int:
        """
        Returns the number of dimensions of the source array.

        Returns:
            The number of dimensions of the source array.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array.dims
            3
        Note:
            The number of dimensions is 3 because the source array is not
            modified.
        """
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        """
        Returns the voxel size of the source array.

        Returns:
            The voxel size of the source array.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array.voxel_size
            Coordinate(x=1.0, y=1.0, z=1.0)
        Note:
            The voxel size is (1.0, 1.0, 1.0) because the source array is not
            modified.
        """
        return self._source_array.voxel_size

    @property
    def roi(self) -> Roi:
        """
        Returns the region of interest of the source array.

        Returns:
            The region of interest of the source array.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array.roi
            Roi(offset=(0, 0, 0), shape=(10, 10, 10))
        Note:
            The region of interest is (0, 0, 0) with shape (10, 10, 10)
            because the source array is not modified.
        """
        return self.crop_roi.intersect(self._source_array.roi)

    @property
    def writable(self) -> bool:
        """
        Returns whether the array is writable.

        Returns:
            False
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array.writable
            False
        Note:
            The array is not writable because it is a virtual array created by
            modifying another array on demand.
        """
        return False

    @property
    def dtype(self):
        """
        Returns the data type of the source array.

        Returns:
            The data type of the source array.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array.dtype
            numpy.dtype('uint8')
        Note:
            The data type is uint8 because the source array is not modified.
        """
        return self._source_array.dtype

    @property
    def num_channels(self) -> int:
        """
        Returns the number of channels of the source array.

        Returns:
            The number of channels of the source array.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array.num_channels
            1
        Note:
            The number of channels is 1 because the source array is not
            modified.
        """
        return self._source_array.num_channels

    @property
    def data(self):
        """
        Returns the data of the source array.

        Returns:
            The data of the source array.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array.data
            array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0, 0, 0, 0, 0
        """
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    @property
    def channels(self):
        """
        Returns the channels of the source array.

        Returns:
            The channels of the source array.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array.channels
            1
        Note:
            The channels is 1 because the source array is not modified.
        """
        return self._source_array.channels

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Returns the data of the source array within the region of interest.

        Args:
            roi: The region of interest.
        Returns:
            The data of the source array within the region of interest.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> array_config = ArrayConfig(
            ...     name='array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array[Roi((0, 0, 0), (5, 5, 5))]
            array([[[
        Note:
            The data is the same as the source array because the source array
            is not modified.
        """
        assert self.roi.contains(roi)
        return self._source_array[roi]

    def _can_neuroglance(self):
        """
        Returns whether the source array can be viewed in Neuroglancer.

        Returns:
            Whether the source array can be viewed in Neuroglancer.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> source_array_config = ArrayConfig(
            ...     name='source_array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array._can_neuroglance()
            False
        Note:
            The source array cannot be viewed in Neuroglancer because the
            source array is not modified.
        """
        return self._source_array._can_neuroglance()

    def _neuroglancer_source(self):
        """
        Returns the source of the source array for Neuroglancer.

        Returns:
            The source of the source array for Neuroglancer.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> source_array_config = ArrayConfig(
            ...     name='source_array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array._neuroglancer_source()
            {'source': 'source_array'}
        Note:
            The source is the source array because the source array is not
            modified.
        """
        return self._source_array._neuroglancer_source()

    def _neuroglancer_layer(self):
        """
        Returns the layer of the source array for Neuroglancer.

        Returns:
            The layer of the source array for Neuroglancer.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> source_array_config = ArrayConfig(
            ...     name='source_array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array._neuroglancer_layer()
            {'source': 'source_array', 'type': 'image'}
        Note:
            The layer is an image because the source array is not modified.
        """
        return self._source_array._neuroglancer_layer()

    def _source_name(self):
        """
        Returns the name of the source array.

        Returns:
            The name of the source array.
        Raises:
            ValueError: If the region of interest to crop to is not within the
                region of interest of the source array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import CropArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> source_array_config = ArrayConfig(
            ...     name='source_array',
            ...     source_array_config=source_array_config,
            ...     roi=Roi((0, 0, 0), (10, 10, 10))
            ... )
            >>> crop_array = CropArray(array_config)
            >>> crop_array._source_name()
            'source_array'
        Note:
            The name is the source array because the source array is not
            modified.
        """
        return self._source_array._source_name()

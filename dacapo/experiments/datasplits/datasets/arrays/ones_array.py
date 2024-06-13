from .array import Array

from funlib.geometry import Roi

import numpy as np

import logging

logger = logging.getLogger(__name__)


class OnesArray(Array):
    """
    This is a wrapper around another `source_array` that simply provides ones
    with the same metadata as the `source_array`.

    This is useful for creating a mask array that is the same size as the
    original array, but with all values set to 1.

    Attributes:
        source_array: The source array that this array is based on.
    Methods:
        like: Create a new OnesArray with the same metadata as another array.
        attrs: Get the attributes of the array.
        axes: Get the axes of the array.
        dims: Get the dimensions of the array.
        voxel_size: Get the voxel size of the array.
        roi: Get the region of interest of the array.
        writable: Check if the array is writable.
        data: Get the data of the array.
        dtype: Get the data type of the array.
        num_channels: Get the number of channels of the array.
        __getitem__: Get a subarray of the array.
    Note:
        This class is not meant to be instantiated directly. Instead, use the
        `like` method to create a new OnesArray with the same metadata as
        another array.
    """

    def __init__(self, array_config):
        """
        Initialize the OnesArray with the given array configuration.

        Args:
            array_config: The configuration of the source array.
        Raises:
            RuntimeError: If the source array is not specified in the
                configuration.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> source_array_config = ArrayConfig(source_array)
            >>> ones_array = OnesArray(source_array_config)
            >>> ones_array.source_array
            NumpyArray(data=array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]), voxel_size=(1.0, 1.0, 1.0), roi=Roi((0, 0, 0), (10, 10, 10)), num_channels=1)
        Notes:
            This class is not meant to be instantiated directly. Instead, use the
            `like` method to create a new OnesArray with the same metadata as
            another array.
        """
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

    @classmethod
    def like(cls, array: Array):
        """
        Create a new OnesArray with the same metadata as another array.

        Args:
            array: The source array.
        Returns:
            The new OnesArray with the same metadata as the source array.
        Raises:
            RuntimeError: If the source array is not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray.like(source_array)
            >>> ones_array.source_array
            NumpyArray(data=array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]), voxel_size=(1.0, 1.0, 1.0), roi=Roi((0, 0, 0), (10, 10, 10)), num_channels=1)
        Notes:
            This class is not meant to be instantiated directly. Instead, use the
            `like` method to create a new OnesArray with the same metadata as
            another array.

        """
        instance = cls.__new__(cls)
        instance._source_array = array
        return instance

    @property
    def attrs(self):
        """
        Get the attributes of the array.

        Returns:
            An empty dictionary.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> ones_array.attrs
            {}
        Notes:
            This method is used to get the attributes of the array. The attributes
            are stored as key-value pairs in a dictionary. This method returns an
            empty dictionary because the OnesArray does not have any attributes.
        """
        return dict()

    @property
    def source_array(self) -> Array:
        """
        Get the source array that this array is based on.

        Returns:
            The source array.
        Raises:
            RuntimeError: If the source array is not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> ones_array.source_array
            NumpyArray(data=array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]), voxel_size=(1.0, 1.0, 1.0), roi=Roi((0, 0, 0), (10, 10, 10)), num_channels=1)
        Notes:
            This method is used to get the source array that this array is based on.
            The source array is the array that the OnesArray is created from. This
            method returns the source array that was specified when the OnesArray
            was created.
        """
        return self._source_array

    @property
    def axes(self):
        """
        Get the axes of the array.

        Returns:
            The axes of the array.
        Raises:
            RuntimeError: If the axes are not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> ones_array.axes
            'zyx'
        Notes:
            This method is used to get the axes of the array. The axes are the
            order of the dimensions of the array. This method returns the axes of
            the array that was specified when the OnesArray was created.
        """
        return self.source_array.axes

    @property
    def dims(self):
        """
        Get the dimensions of the array.

        Returns:
            The dimensions of the array.
        Raises:
            RuntimeError: If the dimensions are not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> ones_array.dims
            (10, 10, 10)
        Notes:
            This method is used to get the dimensions of the array. The dimensions
            are the size of the array along each axis. This method returns the
            dimensions of the array that was specified when the OnesArray was created.
        """
        return self.source_array.dims

    @property
    def voxel_size(self):
        """
        Get the voxel size of the array.

        Returns:
            The voxel size of the array.
        Raises:
            RuntimeError: If the voxel size is not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> ones_array.voxel_size
            (1.0, 1.0, 1.0)
        Notes:
            This method is used to get the voxel size of the array. The voxel size
            is the size of each voxel in the array. This method returns the voxel
            size of the array that was specified when the OnesArray was created.
        """
        return self.source_array.voxel_size

    @property
    def roi(self):
        """
        Get the region of interest of the array.

        Returns:
            The region of interest of the array.
        Raises:
            RuntimeError: If the region of interest is not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> ones_array.roi
            Roi((0, 0, 0), (10, 10, 10))
        Notes:
            This method is used to get the region of interest of the array. The
            region of interest is the region of the array that contains the data.
            This method returns the region of interest of the array that was specified
            when the OnesArray was created.
        """
        return self.source_array.roi

    @property
    def writable(self) -> bool:
        """
        Check if the array is writable.

        Returns:
            False.
        Raises:
            RuntimeError: If the writability of the array is not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> ones_array.writable
            False
        Notes:
            This method is used to check if the array is writable. An array is
            writable if it can be modified in place. This method returns False
            because the OnesArray is read-only and cannot be modified.
        """
        return False

    @property
    def data(self):
        """
        Get the data of the array.

        Returns:
            The data of the array.
        Raises:
            RuntimeError: If the data is not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> ones_array.data
            array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])
        Notes:
            This method is used to get the data of the array. The data is the
            values that are stored in the array. This method returns a subarray
            of the array with all values set to 1.
        """
        raise RuntimeError("Cannot get writable version of this data!")

    @property
    def dtype(self):
        """
        Get the data type of the array.

        Returns:
            The data type of the array.
        Raises:
            RuntimeError: If the data type is not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> ones_array.dtype
            <class 'numpy.bool_'>
        Notes:
            This method is used to get the data type of the array. The data type
            is the type of the values that are stored in the array. This method
            returns the data type of the array that was specified when the OnesArray
            was created.
        """
        return bool

    @property
    def num_channels(self):
        """
        Get the number of channels of the array.

        Returns:
            The number of channels of the array.
        Raises:
            RuntimeError: If the number of channels is not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> ones_array.num_channels
            1
        Notes:
            This method is used to get the number of channels of the array. The
            number of channels is the number of values that are stored at each
            voxel in the array. This method returns the number of channels of the
            array that was specified when the OnesArray was created.
        """
        return self.source_array.num_channels

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Get a subarray of the array.

        Args:
            roi: The region of interest.
        Returns:
            A subarray of the array with all values set to 1.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import OnesArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = OnesArray(source_array)
            >>> roi = Roi((0, 0, 0), (10, 10, 10))
            >>> ones_array[roi]
            array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])
        Notes:
            This method is used to get a subarray of the array. The subarray is
            specified by the region of interest. This method returns a subarray
            of the array with all values set to 1.
        """
        logger.warning("OnesArray is deprecated. Use ConstantArray instead.")
        return np.ones_like(self.source_array.__getitem__(roi), dtype=bool)

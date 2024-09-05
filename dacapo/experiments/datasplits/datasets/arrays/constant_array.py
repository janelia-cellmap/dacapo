from .array import Array

from funlib.geometry import Roi

import numpy as np
import neuroglancer


class ConstantArray(Array):
    """
    This is a wrapper around another `source_array` that simply provides constant value
    with the same metadata as the `source_array`.

    This is useful for creating a mask array that is the same size as the
    original array, but with all values set to 1.

    Attributes:
        source_array: The source array that this array is based on.
    Methods:
        like: Create a new ConstantArray with the same metadata as another array.
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
        `like` method to create a new ConstantArray with the same metadata as
        another array.
    """

    def __init__(self, array_config):
        """
        Initialize the ConstantArray with the given array configuration.

        Args:
            array_config: The configuration of the source array.
        Raises:
            RuntimeError: If the source array is not specified in the
                configuration.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> source_array_config = ArrayConfig(source_array)
            >>> ones_array = ConstantArray(source_array_config)
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
            `like` method to create a new ConstantArray with the same metadata as
            another array.
        """
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )
        self._constant = array_config.constant

    @classmethod
    def like(cls, array: Array):
        """
        Create a new ConstantArray with the same metadata as another array.

        Args:
            array: The source array.
        Returns:
            The new ConstantArray with the same metadata as the source array.
        Raises:
            RuntimeError: If the source array is not specified.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray.like(source_array)
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
            `like` method to create a new ConstantArray with the same metadata as
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
            >>> ones_array.attrs
            {}
        Notes:
            This method is used to get the attributes of the array. The attributes
            are stored as key-value pairs in a dictionary. This method returns an
            empty dictionary because the ConstantArray does not have any attributes.
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
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
            The source array is the array that the ConstantArray is created from. This
            method returns the source array that was specified when the ConstantArray
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
            >>> ones_array.axes
            'zyx'
        Notes:
            This method is used to get the axes of the array. The axes are the
            order of the dimensions of the array. This method returns the axes of
            the array that was specified when the ConstantArray was created.
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
            >>> ones_array.dims
            (10, 10, 10)
        Notes:
            This method is used to get the dimensions of the array. The dimensions
            are the size of the array along each axis. This method returns the
            dimensions of the array that was specified when the ConstantArray was created.
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
            >>> ones_array.voxel_size
            (1.0, 1.0, 1.0)
        Notes:
            This method is used to get the voxel size of the array. The voxel size
            is the size of each voxel in the array. This method returns the voxel
            size of the array that was specified when the ConstantArray was created.
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
            >>> ones_array.roi
            Roi((0, 0, 0), (10, 10, 10))
        Notes:
            This method is used to get the region of interest of the array. The
            region of interest is the region of the array that contains the data.
            This method returns the region of interest of the array that was specified
            when the ConstantArray was created.
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
            >>> ones_array.writable
            False
        Notes:
            This method is used to check if the array is writable. An array is
            writable if it can be modified in place. This method returns False
            because the ConstantArray is read-only and cannot be modified.
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
            >>> ones_array.dtype
            <class 'numpy.bool_'>
        Notes:
            This method is used to get the data type of the array. The data type
            is the type of the values that are stored in the array. This method
            returns the data type of the array that was specified when the ConstantArray
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
            >>> ones_array.num_channels
            1
        Notes:
            This method is used to get the number of channels of the array. The
            number of channels is the number of values that are stored at each
            voxel in the array. This method returns the number of channels of the
            array that was specified when the ConstantArray was created.
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
            >>> from dacapo.experiments.datasplits.datasets.arrays import ConstantArray
            >>> from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
            >>> from funlib.geometry import Roi
            >>> import numpy as np
            >>> source_array = NumpyArray(np.zeros((10, 10, 10)))
            >>> ones_array = ConstantArray(source_array)
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
        return (
            np.ones_like(self.source_array.__getitem__(roi), dtype=bool)
            * self._constant
        )

    def _can_neuroglance(self):
        """
        This method returns True if the source array can be visualized in neuroglance.

        Returns:
            bool: True if the source array can be visualized in neuroglance.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array._can_neuroglance()
        Note:
            This method is used to return True if the source array can be visualized in neuroglance.
        """
        return True

    def _neuroglancer_source(self):
        """
        This method returns the source array for neuroglancer.

        Returns:
            neuroglancer.LocalVolume: The source array for neuroglancer.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array._neuroglancer_source()
        Note:
            This method is used to return the source array for neuroglancer.
        """
        # return self._source_array._neuroglancer_source()
        shape = self.source_array[self.source_array.roi].shape[-3:]
        return np.ones(shape, dtype=np.uint64) * self._constant

    def _combined_neuroglancer_source(self) -> neuroglancer.LocalVolume:
        """
        Combines dimensions and metadata from self._source_array._neuroglancer_source()
        with data from self._neuroglancer_source().

        Returns:
            neuroglancer.LocalVolume: The combined neuroglancer source.
        """
        source_array_volume = self._source_array._neuroglancer_source()
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
        return neuroglancer.SegmentationLayer(
            source=self._combined_neuroglancer_source()
        )

    def _source_name(self):
        """
        This method returns the name of the source array.

        Returns:
            str: The name of the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array._source_name()
        Note:
            This method is used to return the name of the source array.
        """
        # return self._source_array._source_name()
        return f"{self._constant}_of_{self.source_array._source_name()}"

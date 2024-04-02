from .array import Array

from funlib.geometry import Coordinate, Roi


import neuroglancer

import numpy as np


class SumArray(Array):
    """
    This class provides a sum array. This array is a virtual array that is created by summing
    multiple source arrays. The source arrays must have the same shape and ROI. 

    Attributes:
        name: str
            The name of the array.
        _source_arrays: List[Array]
            The source arrays to sum.
        _source_array: Array
            The first source array.
    Methods:
        __getitem__(roi: Roi) -> np.ndarray
            Get the data for the given region of interest.
        _can_neuroglance() -> bool
            Check if neuroglance can be used.
        _neuroglancer_source() -> Dict
            Return the source for neuroglance.
        _neuroglancer_layer() -> Tuple[neuroglancer.SegmentationLayer, Dict]
            Return the neuroglancer layer.
        _source_name() -> str
            Return the source name.
    Note:
        This class is a subclass of Array.
    """

    def __init__(self, array_config):
        """
        Initialize the SumArray.

        Args:
            array_config: SumArrayConfig
                The configuration for the sum array.
        Returns:
            SumArray: The sum array.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays.sum_array import SumArray
            >>> from dacapo.experiments.datasplits.datasets.arrays.sum_array_config import SumArrayConfig
            >>> from dacapo.experiments.datasplits.datasets.arrays.tiff_array import TiffArray
            >>> from dacapo.experiments.datasplits.datasets.arrays.tiff_array_config import TiffArrayConfig
            >>> from funlib.geometry import Coordinate
            >>> from pathlib import Path
            >>> sum_array = SumArray(SumArrayConfig(name="sum", source_array_configs=[TiffArrayConfig(file_name=Path("data.tiff"), offset=Coordinate([0, 0, 0]), voxel_size=Coordinate([1, 1, 1]), axes=["x", "y", "z"])]))
        Note:
            This class is a subclass of Array.
    
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
        The axes of the array.

        Returns:
            List[str]: The axes of the array.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array.axes
            ['x', 'y', 'z']
        Note:
            This class is a subclass of Array.
        """
        return [x for x in self._source_array.axes if x != "c"]

    @property
    def dims(self) -> int:
        """
        The number of dimensions of the array.

        Returns:
            int: The number of dimensions of the array.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array.dims
            3
        Note:
            This class is a subclass of Array.
        """
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        """
        The size of each voxel in each dimension.

        Returns:
            Coordinate: The size of each voxel in each dimension.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array.voxel_size
            Coordinate([1, 1, 1])
        Note:
            This class is a subclass of Array.
        """
        return self._source_array.voxel_size

    @property
    def roi(self) -> Roi:
        """
        The region of interest of the array.

        Args:
            roi: Roi
                The region of interest.
        Returns:
            Roi: The region of interest.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array.roi
            Roi(Coordinate([0, 0, 0]), Coordinate([100, 100, 100]))
        Note:
            This class is a subclass of Array.
        """
        return self._source_array.roi

    @property
    def writable(self) -> bool:
        """
        Check if the array is writable.

        Args:
            writable: bool
                Check if the array is writable.
        Returns:
            bool: True if the array is writable, otherwise False.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array.writable
            False
        Note:
            This class is a subclass of Array.
        """
        return False

    @property
    def dtype(self):
        """
        The data type of the array.

        Args:
            dtype: np.uint8
                The data type of the array.
        Returns:
            np.uint8: The data type of the array.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array.dtype
            np.uint8
        Note:
            This class is a subclass of Array.

        """
        return np.uint8

    @property
    def num_channels(self):
        """
        The number of channels in the array.

        Args:
            num_channels: Optional[int]
                The number of channels in the array.
        Returns:
            Optional[int]: The number of channels in the array.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array.num_channels
            None
        Note:
            This class is a subclass of Array.

        """
        return None

    @property
    def data(self):
        """
        Get the data of the array.

        Args:
            data: np.ndarray
                The data of the array.
        Returns:
            np.ndarray: The data of the array.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array.data
            np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        Note:
            This class is a subclass of Array.
        """
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    @property
    def attrs(self):
        """
        Return the attributes of the array.

        Args:
            attrs: Dict
                The attributes of the array.
        Returns:
            Dict: The attributes of the array.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array.attrs
            {}
        Note:
            This class is a subclass of Array.
        """
        return self._source_array.attrs

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Get the data for the given region of interest.

        Args:
            roi: Roi
                The region of interest.
        Returns:
            np.ndarray: The data for the given region of interest.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array[roi]
            np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        Note:  
            This class is a subclass of Array.
        """
        return np.sum(
            [source_array[roi] for source_array in self._source_arrays], axis=0
        )

    def _can_neuroglance(self):
        """
        Check if neuroglance can be used.

        Args:
            can_neuroglance: bool
                Check if neuroglance can be used.
        Returns:
            bool: True if neuroglance can be used, otherwise False.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array._can_neuroglance()
            False
        Note:
            This class is a subclass of Array.
        """
        return self._source_array._can_neuroglance()

    def _neuroglancer_source(self):
        """
        Return the source for neuroglance.

        Args:
            source: Dict
                The source for neuroglance.
        Returns:
            Dict: The source for neuroglance.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array._neuroglancer_source()
            {'source': 'precomputed://https://mybucket/segmentation', 'type': 'segmentation', 'voxel_size': [1, 1, 1]}
        Note:
            This class is a subclass of Array.

        """
        return self._source_array._neuroglancer_source()

    def _neuroglancer_layer(self):
        """
        Return the neuroglancer layer.

        Args:
            layer: Tuple[neuroglancer.SegmentationLayer, Dict]
                The neuroglancer layer.
        Returns:
            Tuple[neuroglancer.SegmentationLayer, Dict]: The neuroglancer layer.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array._neuroglancer_layer()
            (SegmentationLayer(source={'source': 'precomputed://https://mybucket/segmentation', 'type': 'segmentation', 'voxel_size': [1, 1, 1]}, visible=False), {})
        Note:
            This class is a subclass of Array.

        """
        # Generates an Segmentation layer

        layer = neuroglancer.SegmentationLayer(source=self._neuroglancer_source())
        kwargs = {
            "visible": False,
        }
        return layer, kwargs

    def _source_name(self):
        """
        Return the source name.

        Args:
            source_name: str
                The source name.
        Returns:
            str: The source name.
        Raises:
            ValueError:
                Cannot get a writable view of this array because it is a virtual array created by modifying another array on demand.
        Examples:
            >>> sum_array._source_name()
            'data.tiff'
        Note:
            This class is a subclass of Array.

        """
        return self._source_array._source_name()

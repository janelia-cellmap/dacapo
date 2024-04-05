from .array import Array

from funlib.geometry import Coordinate, Roi

import neuroglancer

import numpy as np


class BinarizeArray(Array):
    """
    This is wrapper around a ZarrArray containing uint annotations.
    Because we often want to predict classes that are a combination
    of a set of labels we wrap a ZarrArray with the BinarizeArray
    and provide something like `groupings=[("mito", [3,4,5])]`
    where 4 corresponds to mito_mem (mitochondria membrane), 5 is mito_ribo 
    (mitochondria ribosomes), and 3 is everything else that is part of a 
    mitochondria. The BinarizeArray will simply combine labels 3,4,5 into 
    a single binary channel for the class of "mito".

    We use a single channel per class because some classes may overlap.
    For example if you had `groupings=[("mito", [3,4,5]), ("membrane", [4, 8, 1])]`
    where 4 is mito_mem, 8 is er_mem (ER membrane), and 1 is pm (plasma membrane).
    Now you can have a binary classification for membrane or not which in
    some cases overlaps with the channel for mitochondria which includes
    the mito membrane.

    Attributes:
        name (str): The name of the array.
        source_array (Array): The source array to binarize.
        background (int): The label to treat as background.
        groupings (List[Tuple[str, List[int]]]): A list of tuples where the first
            element is the name of the class and the second element is a list of
            labels that should be combined into a single binary channel.
    Methods:
        __init__(self, array_config): This method initializes the BinarizeArray object.
        __attrs_post_init__(self): This method is called after the instance has been initialized by the constructor. It is used to set the default_config to an instance of ArrayConfig if it is None.
        __getitem__(self, roi: Roi) -> np.ndarray: This method returns the binary channels for the given region of interest.
        _can_neuroglance(self): This method returns True if the source array can be visualized in neuroglance.
        _neuroglancer_source(self): This method returns the source array for neuroglancer.
        _neuroglancer_layer(self): This method returns the neuroglancer layer for the source array.
        _source_name(self): This method returns the name of the source array.
    Note:
        This class is used to create a BinarizeArray object which is a wrapper around a ZarrArray containing uint annotations.
    """

    def __init__(self, array_config):
        """
        This method initializes the BinarizeArray object. 

        Args:
            array_config (ArrayConfig): The array configuration.
        Raises:
            AssertionError: If the source array has channels.
        Examples:
            >>> binarize_array = BinarizeArray(array_config)
        Note:
            This method is used to initialize the BinarizeArray object.
        """
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )
        self.background = array_config.background

        assert (
            "c" not in self._source_array.axes
        ), "Cannot initialize a BinarizeArray with a source array with channels"

        self._groupings = array_config.groupings

    @property
    def attrs(self):
        """
        This method returns the attributes of the source array.

        Returns:
            Dict: The attributes of the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array.attrs
        Note:
            This method is used to return the attributes of the source array.
        """
        return self._source_array.attrs

    @property
    def axes(self):
        """
        This method returns the axes of the source array.

        Returns:
            List[str]: The axes of the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array.axes
        Note:
            This method is used to return the axes of the source array.
        """
        return ["c"] + self._source_array.axes

    @property
    def dims(self) -> int:
        """
        This method returns the dimensions of the source array.

        Returns:
            int: The dimensions of the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array.dims
        Note:
            This method is used to return the dimensions of the source array.
        """
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        """
        This method returns the voxel size of the source array.
        
        Returns:
            Coordinate: The voxel size of the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array.voxel_size
        Note:
            This method is used to return the voxel size of the source array.
        """
        return self._source_array.voxel_size

    @property
    def roi(self) -> Roi:
        """
        This method returns the region of interest of the source array.

        Returns:
            Roi: The region of interest of the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array.roi
        Note:
            This method is used to return the region of interest of the source array.
        """
        return self._source_array.roi

    @property
    def writable(self) -> bool:
        """
        This method returns True if the source array is writable.

        Returns:
            bool: True if the source array is writable.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array.writable
        Note:
            This method is used to return True if the source array is writable.
        """
        return False

    @property
    def dtype(self):
        """
        This method returns the data type of the source array.

        Returns:
            np.dtype: The data type of the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array.dtype
        Note:
            This method is used to return the data type of the source array.
        """
        return np.uint8

    @property
    def num_channels(self) -> int:
        """
        This method returns the number of channels in the source array.

        Returns:
            int: The number of channels in the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array.num_channels
        Note:
            This method is used to return the number of channels in the source array.
        
        """
        return len(self._groupings)

    @property
    def data(self):
        """
        This method returns the data of the source array.

        Returns:
            np.ndarray: The data of the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array.data
        Note:
            This method is used to return the data of the source array.
        """
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    @property
    def channels(self):
        """
        This method returns the channel names of the source array.

        Returns:
            Iterator[str]: The channel names of the source array.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array.channels
        Note:
            This method is used to return the channel names of the source array.
        """
        return (name for name, _ in self._groupings)

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        This method returns the binary channels for the given region of interest.

        Args:
            roi (Roi): The region of interest.
        Returns:
            np.ndarray: The binary channels for the given region of interest.
        Raises:
            ValueError: If the source array is not writable.
        Examples:
            >>> binarize_array[roi]
        Note:
            This method is used to return the binary channels for the given region of interest.
        """
        labels = self._source_array[roi]
        grouped = np.zeros((len(self._groupings), *labels.shape), dtype=np.uint8)
        for i, (_, ids) in enumerate(self._groupings):
            if len(ids) == 0:
                grouped[i] += labels != self.background
            for id in ids:
                grouped[i] += labels == id
        return grouped

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
        return self._source_array._can_neuroglance()

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
        return self._source_array._neuroglancer_source()

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
        layer = neuroglancer.SegmentationLayer(source=self._neuroglancer_source())
        return layer

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
        return self._source_array._source_name()

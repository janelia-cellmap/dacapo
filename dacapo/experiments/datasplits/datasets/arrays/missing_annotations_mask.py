from .array import Array

from funlib.geometry import Coordinate, Roi

from fibsem_tools.metadata.groundtruth import LabelList

import neuroglancer

import numpy as np


class MissingAnnotationsMask(Array):
    """
    This is wrapper around a ZarrArray containing uint annotations.
    Complementary to the BinarizeArray class where we convert labels
    into individual channels for training, we may find crops where a
    specific label is present, but not annotated. In that case you
    might want to avoid training specific channels for specific
    training volumes.
    See package fibsem_tools for appropriate metadata format for indicating
    presence of labels in your ground truth.
    "https://github.com/janelia-cosem/fibsem-tools"

    Attributes:
        array_config: A BinarizeArrayConfig object
    Methods:
        __getitem__(roi: Roi) -> np.ndarray: Returns a binary mask of the
            annotations that are present but not annotated.
    Note:
        This class is not meant to be used directly. It is used by the
        BinarizeArray class to mask out annotations that are present but
        not annotated.
    """

    def __init__(self, array_config):
        """
        Initializes the MissingAnnotationsMask class

        Args:
            array_config (BinarizeArrayConfig): A BinarizeArrayConfig object
        Raises:
            AssertionError: If the source array has channels
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> missing_annotations_mask = MissingAnnotationsMask(MissingAnnotationsMaskConfig(source_array, groupings))
        Notes:
            This is a helper function for the BinarizeArray class
        """
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

        assert (
            "c" not in self._source_array.axes
        ), "Cannot initialize a BinarizeArray with a source array with channels"

        self._groupings = array_config.groupings

    @property
    def axes(self):
        """
        Returns the axes of the source array

        Returns:
            list: Axes of the source array
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array.axes
            ['x', 'y', 'z']
        Notes:
            This is a helper function for the BinarizeArray class
        """
        return ["c"] + self._source_array.axes

    @property
    def dims(self) -> int:
        """
        Returns the number of dimensions of the source array

        Returns:
            int: Number of dimensions of the source array
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array.dims
            3
        Notes:
            This is a helper function for the BinarizeArray class
        """
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        """
        Returns the voxel size of the source array

        Returns:
            Coordinate: Voxel size of the source array
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array.voxel_size
            Coordinate(x=4, y=4, z=40)
        Notes:
            This is a helper function for the BinarizeArray class

        """
        return self._source_array.voxel_size

    @property
    def roi(self) -> Roi:
        """
        Returns the region of interest of the source array

        Returns:
            Roi: Region of interest of the source array
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array.roi
            Roi(offset=(0, 0, 0), shape=(100, 100, 100))
        Notes:
            This is a helper function for the BinarizeArray class
        """
        return self._source_array.roi

    @property
    def writable(self) -> bool:
        """
        Returns whether the source array is writable

        Returns:
            bool: Whether the source array is writable
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array.writable
            False
        Notes:
            This is a helper function for the BinarizeArray class

        """
        return False

    @property
    def dtype(self):
        """
        Returns the data type of the source array

        Returns:
            np.dtype: Data type of the source array
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array.dtype
            np.uint8
        Notes:
            This is a helper function for the BinarizeArray class

        """
        return np.uint8

    @property
    def num_channels(self) -> int:
        """
        Returns the number of channels

        Returns:
            int: Number of channels
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array.num_channels
            2
        Notes:
            This is a helper function for the BinarizeArray class


        """
        return len(self._groupings)

    @property
    def data(self):
        """
        Returns the data of the source array

        Returns:
            np.ndarray: Data of the source array
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array.data
            np.ndarray(...)
        Notes:
            This is a helper function for the BinarizeArray class

        """
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    @property
    def attrs(self):
        """
        Returns the attributes of the source array

        Returns:
            dict: Attributes of the source array
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array.attrs
            {'name': 'source_array', 'resolution': [4, 4, 40]}
        Notes:
            This is a helper function for the BinarizeArray class
        """
        return self._source_array.attrs

    @property
    def channels(self):
        """
        Returns the names of the channels

        Returns:
            Generator[str]: Names of the channels
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array.channels
            Generator['channel1', 'channel2', ...]
        Notes:
            This is a helper function for the BinarizeArray class
        """
        return (name for name, _ in self._groupings)

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Returns a binary mask of the annotations that are present but not annotated.

        Args:
            roi (Roi): Region of interest to get the mask for
        Returns:
            np.ndarray: Binary mask of the annotations that are present but not annotated
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> missing_annotations_mask = MissingAnnotationsMask(MissingAnnotationsMaskConfig(source_array, groupings))
            >>> roi = Roi(...)
            >>> missing_annotations_mask[roi]
            np.ndarray(...)
        Notes:
            - This is a helper function for the BinarizeArray class
            - Number of channels in the mask is equal to the number of groupings
            - Nuclues is a special case where we mask out the whole channel if any of the
              sub-organelles are present but not annotated
        """
        labels = self._source_array[roi]
        grouped = np.ones((len(self._groupings), *labels.shape), dtype=bool)
        grouped[:] = labels > 0
        try:
            labels_list = LabelList.parse_obj({"labels": self.attrs["labels"]}).labels
            present_not_annotated = set(
                [
                    label.value
                    for label in labels_list
                    if label.annotationState.present
                    and not label.annotationState.annotated
                ]
            )
            for i, (_, ids) in enumerate(self._groupings):
                if any([id in present_not_annotated for id in ids]):
                    grouped[i] = 0

        except KeyError:
            pass
        return grouped

    def _can_neuroglance(self):
        """
        Returns whether the array can be visualized in neuroglancer

        Returns:
            bool: Whether the array can be visualized in neuroglancer
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array._can_neuroglance()
            True
        Notes:
            This is a helper function for the neuroglancer layer

        """
        return self._source_array._can_neuroglance()

    def _neuroglancer_source(self):
        """
        Returns a neuroglancer source for the array

        Returns:
            neuroglancer.LocalVolume: Neuroglancer source for the array
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array._neuroglancer_source()
            neuroglancer.LocalVolume(...)
        Notes:
            This is a helper function for the neuroglancer layer
        """
        return self._source_array._neuroglancer_source()

    def _neuroglancer_layer(self):
        """
        Returns a neuroglancer Segmentation layer for the array

        Returns:
            neuroglancer.SegmentationLayer: Segmentation layer for the array
            dict: Keyword arguments for the layer
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array._neuroglancer_layer()
            (neuroglancer.SegmentationLayer, dict)
        Notes:
            This is a helper function for the neuroglancer layer
        """
        # Generates an Segmentation layer

        layer = neuroglancer.SegmentationLayer(source=self._neuroglancer_source())
        kwargs = {
            "visible": False,
        }
        return layer, kwargs

    def _source_name(self):
        """
        Returns the name of the source array

        Returns:
            str: Name of the source array
        Raises:
            ValueError: If the source array does not have a name
        Examples:
            >>> source_array = ZarrArray(ZarrArrayConfig(...))
            >>> source_array._source_name()
            'source_array'
        Notes:
            This is a helper function for the neuroglancer layer name
        """
        return self._source_array._source_name()

class MergeInstancesArray(Array):
    """
    Class for merging different sources into a single array.

    This class merges the source arrays defined in the array configuration. 
    It implements different properties, and methods, to handle the merging process.

    Attributes:
        array_config: Configuration specifying how to initialize the array.
        name: The name of the array.
        _source_arrays: The list of source arrays to be merged based on the source configurations.
        _source_array: The first array from the list of source arrays.
    """
    def __init__(self, array_config):
        """
        Initialize the merge instances array class.

        Args:
            array_config: Configurations of the array to be initialised.
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
        Provide the axes excluding 'c' of the source array.
        
        Returns:
            list: The axes of the source array excluding 'c'.
        """

    @property
    def dims(self) -> int:
        """
        Provide the dimension of the source array.

        Returns:
            int: The dimension of the source array.
        """
        
    @property
    def voxel_size(self) -> Coordinate:
        """
        Provide the voxel size of the source array.

        Returns:
            Coordinate: The voxel size of the source array.
        """
        
    @property
    def roi(self) -> Roi:
        """
        Provide the region of interest (ROI) of the source array.

        Returns:
            Roi: The region of interest of the source array.
        """

    @property
    def writable(self) -> bool:
        """
        Indicate whether the array is writable.

        Returns:
            bool: Always False, indicating non-writable.
        """
        
    @property
    def dtype(self):
        """
        Provide the data type - unsigned integer of 8 bits.

        Returns:
            numpy data type: The data type of the array elements.
        """
        
    @property
    def num_channels(self):
        """
        Number of channels of the array, which is not defined here.

        Returns:
            None.
        """
        
    @property
    def data(self):
        """
        This property is not defined in the current class.

        Raises:
            ValueError: if attempted to retrieve the data property.
        """
        
    @property
    def attrs(self):
        """
        Provide the attributes of the source array.

        Returns:
            dict: The attrs dictionary of the source array.
        """
        
    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Get a subset of the merged array for the specified region of interest (ROI).

        Args:
            roi: The region of interest from the merged array.

        Returns:
            np.ndarray: The merged array for the particular region of interest.
        """
        
    def _can_neuroglance(self):
        """
        Check if the source array can be visualized with neuroglancer.

        Returns:
            bool: True if neuroglancer can visualize the source array, False otherwise.
        """
        
    def _neuroglancer_source(self):
        """
        Provide the source of the neuroglancer visualization.

        Returns:
            object: Source of the neuroglancer visualization.
        """
        
    def _neuroglancer_layer(self):
        """
        Generate a Segmentation layer for neuroglancer visualization. 
        
        Returns:
            layer: The neuroglancer SegmentationLayer object.
            kwargs: A dictionary of keyword arguments (visible is always set as False).
        """
        
    def _source_name(self):
        """
        Provide the name of the source array.

        Returns:
            str: Name of the source array
        """

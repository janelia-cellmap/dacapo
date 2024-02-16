```python
class SumArray(Array):
    """
    SumArray is a subclass of the class Array. It represents a virtual array that 
    does not support writing. The values of the array are computed on demand by 
    summing the values of the source arrays.
    
    Attributes:
        name: str: Name of the array.
        _source_array: Array: The first source array in the list of source arrays.
        _source_arrays: list: The source arrays that are summed to produce this array.
    """

    def __init__(self, array_config):
        """
        Initializes the SumArray with the specified array_config.

        Args:
            array_config: The configuration for this array.
        """
        
    @property
    def axes(self):
        """
        Returns a list of axes excluding the 'c' axis.
        
        Returns:
            list: List of axes.
        """
        
    @property
    def dims(self) -> int:
        """
        Returns the dimensions of the source array.

        Returns:
            int: Number of dimensions.
        """
        
    @property
    def voxel_size(self) -> Coordinate:
        """
        Returns the size of the voxels in the source array.

        Returns:
            Coordinate: Voxel size.
        """
        
    @property
    def roi(self) -> Roi:
        """
        Returns the Roi of the source array.

        Returns:
            Roi: Region Of Interest.
        """
        
    @property
    def writable(self) -> bool:
        """
        Indicates whether the array is writable or not.
        
        Returns:
            bool: False, as this is a virtual array.
        """
        
    @property
    def dtype(self):
        """
        Returns the data type of the array.
        
        Returns:
            dtype: Data type of the array.
        """
        
    @property
    def num_channels(self):
        """
        Get the number of channels for this array
        
        Returns:
            None: as this function is not currently implemented.
        """
        
    @property
    def attrs(self):
        """
        Returns the attributes of the source array.
        
        Returns:
            dict: attribute dictionary of the source array.
        """
        
    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Returns the sum of the values in the specified region of interest.

        Args:
            roi: Region of interest.

        Returns:
            ndarray: The summed values.
        """
        
    def _can_neuroglance(self):
        """
        Determines if the soure array can neuroglance.
        
        Returns:
            bool: True if source array can neuroglance, else False.
        """
        
    def _neuroglancer_source(self):
        """
        Returns the neuroglancer source of the source array.
        
        Returns:
            Neuroglancer source of the source array.
        """

    def _neuroglancer_layer(self):
        """
        Generates a segmentation layer with a neuroglancer source.
        
        Returns:
            tuple: The segmentation layer.
        """
        
    def _source_name(self):
        """
        Returns the source name of the source array.
        
        Returns:
            str: The source name. 
        """
```

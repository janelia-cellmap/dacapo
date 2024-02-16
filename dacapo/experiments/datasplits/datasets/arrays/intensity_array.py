```python
from .array import Array

from funlib.geometry import Coordinate, Roi

import numpy as np


class IntensitiesArray(Array):
    """
    A class used to represent an Intensities Array.
    This is a wrapper for another array that will normalize intensities to
    the range (0, 1) and convert to float32. This class is particularly
    useful if your intensities are stored as uint8 or similar, 
    and your model requires floats as input.
    
    Args:
        array_config (Array): An array of configuration parameters.
    """

    def __init__(self, array_config):
        """
        Initializes IntensitiesArray with array configuration.
        """
        ...

    @property
    def attrs(self):
        """
        Returns attribute of source array.
        """
        ...

    @property
    def axes(self):
        """
        Returns axes of source array.
        """
        ...

    @property
    def dims(self) -> int:
        """
        Returns dimensions of source array.
        
        Returns:
            int: Dimensions of the source array.
        """
        ...

    @property
    def voxel_size(self) -> Coordinate:
        """
        Returns size of voxel of source array.
        
        Returns:
            Coordinate: Size of voxel of the source array.
        """
        ...

    @property
    def roi(self) -> Roi:
        """
        Returns region of interest (roi) of source array.
        
        Returns:
            Roi: Region of interest (roi) of the source array.
        """
        ...

    @property
    def writable(self) -> bool:
        """
        Checks if source array can be overwritten. 

        Returns:
            bool: False, as source array can't be modified.
        """
        ...

    @property
    def dtype(self):
        """
        Returns type of data present in source array.
        
        Returns:
             dtype: Data type which is always float32. 
        """
        ...

    @property
    def num_channels(self) -> int:
        """
        Returns number of channels of source array.
        
        Returns:
            int: Number of channels of the source array.
        """
        ...

    @property
    def data(self):
        """
        Raises ValueError if called, as no writable view of array is available.
        """
        ...

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Returns normalized intensities.

        Takes ROI as input, calculates normalized intensity and returns.
        
        Args:
            roi (Roi): Region of interest.
        
        Returns:
            np.ndarray: Normalized intensities corresponding to ROI.
        """
        ...

    def _can_neuroglance(self):
        """
        Checks if source array can be visualised using neuroglancer.

        Returns:
            bool: True if source array is compatible with neuroglancer, False otherwise.
        """
        ...

    def _neuroglancer_layer(self):
        """
        Returns the neuroglancer layer of source array.

        Returns:
            dict: Detailing the layers in neuroglancer. 
        """
        ...

    def _source_name(self):
        """
        Returns the source name of the array.
        
        Returns:
            str: Source name of the array.
        """
        ...
```
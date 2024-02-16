"""
The `NumpyArray` class is a wrapper for a numpy array to make it compatible with the DaCapo Array interface.

Attributes:
    _data (np.ndarray): Underlying data of the Array.
    _dtype (np.dtype): Data type of the elements in the array.
    _roi (Roi): Region of interest within the Array.
    _voxel_size (Coordinate): Size of a voxel in the Array.
    _axes (List[str]): Axes of the data.

Methods:

__init__: This function is not intended to be used as it raises a RuntimeError. The Array should
          be created with the `from_gp_array` or `from_np_array` classmethods.

attrs: Returns an empty dictionary. This property is kept for compatibility with Gunpowder Arrays.

from_gp_array: Creates a NumpyArray from a gunpowder array.

from_np_array: Creates a NumpyArray from a numpy array.

axes: Returns a list of strings representing the axes of the Array.

dims: Returns the number of dimensions in the Region of Interest.

voxel_size: Returns the voxel size of the Array.

roi: Returns the region of interest of the Array.

writable: Always returns True. Indicates that the array data can be modified.

data: Returns the underlying numpy array.

dtype: Returns the data type of the elements in the array.

num_channels: Returns the number of channels in the array data, otherwise returns None.

"""

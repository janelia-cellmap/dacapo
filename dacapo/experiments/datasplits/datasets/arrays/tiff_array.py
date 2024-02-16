"""
A Python class designed to handles tiff array.

This class `TiffArray` inherits properties and methods from `Array` class but it specifically works for tiff array.
It uses existing libraries i.e, funlib.geometry, lazy_property, tifffile, logging and pathlib.
And has data properties to store metadata type information about tiff files.

Attributes:
    _offset: A Coordinate from funlib.geometry, which represents the positioning offset of the tiff image.
    _file_name: A Path object from pathlib, which represents the path to the Tiff file.
    _voxel_size: A Coordinate from funlib.geometry, which represents the voxel size of the tiff image.
    _axes: A list of strings, which is used to maintain axes information.

Methods:
    attrs: Property method, not yet implemented.
    axes: Returns the axes of the TiffArray.
    dims: Returns the dimensions of the voxel size.
    shape: Returns the spatial shape of the TiffArray data.
    voxel_size: Returns the voxel size of the TiffArray.
    roi: Returns the region of interest (Roi) for the Tiff Array data.
    writable: Returns a boolean indicating whether the TiffArray can be modified or not.
    dtype: Returns the data type of TiffArray data.
    num_channels: Returns the number of channels in the TiffArray if available.
    spatial_axes: Returns the spatial axes of the TiffArray excluding channel 'c'.
    data: Returns values from the actual Tiff file.
"""
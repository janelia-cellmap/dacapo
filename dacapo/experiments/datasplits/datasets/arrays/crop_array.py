"""
The CropArray class extends Array class and it allows to crop a larger array to a smaller array based on a region of interest (ROI). This class is specifically designed for handling three-dimensional image analysis tasks. CropArray class attributes and methods allow precise control over the array data and properties. 

Attributes:
    _source_array : Array
        The original large array from which a smaller array is derived.
    name : str
        Name of the array.
    crop_roi: Roi
        The region of interest that defines the portion of the larger array to form the smaller array.
    attrs:
        Gets the attributes from the source array.
    axes:
        Gets the axis info from the source array.
    dims : int
        Gets the dimensions from the source array.
    voxel_size: Coordinate
        Gets the voxel size from the source array.
    roi : Roi
        The ROI that is the intersection of the crop_roi and the source array's roi.
    writable : 
        Returns False as the cropped array is not writable. 
    dtype:
        Gets the data type from the source array.
    num_channels: int
        Gets the number of channels from the source array.
    data:
        Raises error as the source array is a virtual array that is created by modifying another array on demand.
    channels:
        Gets the channels info from the source array.

Methods:
    __getitem__(self, roi: Roi) -> np.ndarray:
        Returns the contents of the array for the supplied ROI.
    _can_neuroglance(self):
        Checks if _source_array can be used for neuroglance visualization.
    _neuroglancer_source(self):
        Gets the neuroglancer source from _source_array.
    _neuroglancer_layer(self):
        Gets the neuroglancer layer from _source_array.
    _source_name(self):
        Gets the source name from _source_array.
"""
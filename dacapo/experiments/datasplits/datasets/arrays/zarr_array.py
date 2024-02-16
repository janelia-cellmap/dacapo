"""
ZarrArray Class
---------------
This class implements the Array class, and its purpose is to interact with larger-than-memory 
computational datasets. It allows you to grow, shrink, slice, chop, filter, transform and classify datasets. 

Attributes:
----------
name : string
    The name of the ZarrArray object.

file_name : str
    The path to the ZarrArray file.

dataset : Array
    The dataset which is included in the file.

_attrs : Attributes
    The attributes associated with the ZarrArray object.
    
_axes : list
    The axes of the zarr array.

snap_to_grid : [type]
    A signifier of how the ZArrArray is snap to a grid.

properties:
----------
voxel_size : Coordinate
    Returns the voxel dimensions of the data.

roi : Roi
    Returns the Roi object which is associated with the dataset.

writable : bool
    Returns True because the data are always writable.

dtype : data-type
    Returns data type of the array's elements.

num_channels : int, Optional
    Returns the number of channels if 'c' is present in axes.

spatial_axes : List[str]
    Returns the list of spatial axes in the array.

data : Any
    Returns the data in the array.

Methods:
----------
__getitem__() : Returns the item at the specified index.

__setitem__() : Sets an item at the specified index.

create_from_array_identifier() : Creates a new ZarrArray from an array identifier.

open_from_array_identifier() : Opens the ZarrArray and returns instance.

_can_neuroglance() : Returns if the class can use neuroglancer or not.

_neuroglancer_source() : Returns source type based on the file name.

_neuroglancer_layer() : Generates an Image layer.

_transform_matrix() : Returns a transformation matrix based on the file name.

_output_dimensions() : Returns output dimensions of an array.

_source_name() : It returns object name.


add_metadata(metadata: Dict[str, Any]) 
    Adds metadata to the ZarrArray dataset.
"""

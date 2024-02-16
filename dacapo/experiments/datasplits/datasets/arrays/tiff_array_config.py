import attr

from .array_config import ArrayConfig
from .tiff_array import TiffArray

from funlib.geometry import Coordinate

from pathlib import Path
from typing import List


@attr.s
class ZarrArrayConfig(ArrayConfig):
    """
    A configuration class for zarr array setup and manipulation.

    This class extends the ArrayConfig base class and is responsible for setting
    up the configuration for the TiffArray type. This includes the file name of the 
    zarr container, an offset for alignment with other arrays, the voxel dimensions 
    and the axes of the array.

    Attributes:
        array_type: An attribute representing TiffArray type disposition.
        file_name (Path): The filename of the zarr container being regulated.
        offset (Coordinate): The offset for aligning this array with other arrays.
        voxel_size (Coordinate): The size of each voxel in each dimension.
        axes (List[str]): The axes of the particular array in use.
    """
    
    array_type = TiffArray

    file_name: Path = attr.ib(
        metadata={"help_text": "The file name of the zarr container."}
    )
    offset: Coordinate = attr.ib(
        metadata={
            "help_text": "The offset for this array. This must be provided "
            "to align this array with any others provided."
        }
    )
    voxel_size: Coordinate = attr.ib(
        metadata={"help_text": "The size of each voxel in each dimension."}
    )
    axes: List[str] = attr.ib(metadata={"help_text": "The axes of your array"})
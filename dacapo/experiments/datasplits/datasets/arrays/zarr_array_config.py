import attr

from .array_config import ArrayConfig

from funlib.geometry import Coordinate
from funlib.persistence import open_ds

from upath import UPath as Path

from typing import Optional, List, Tuple


@attr.s
class ZarrArrayConfig(ArrayConfig):
    """
    This config class provides the necessary configuration for a zarr array.

    A zarr array is a container for large, multi-dimensional arrays. It is similar to HDF5, but is designed to work
    with large arrays that do not fit into memory. Zarr arrays can be stored on disk or in the cloud
    and can be accessed concurrently by multiple processes. Zarr arrays can be compressed and
    support chunked, N-dimensional arrays.

    Attributes:
        file_name: Path
            The file name of the zarr container.
        dataset: str
            The name of your dataset. May include '/' characters for nested heirarchies
        snap_to_grid: Optional[Coordinate]
            If you need to make sure your ROI's align with a specific voxel_size
        _axes: Optional[List[str]]
            The axis_names of your data!
    Methods:
        verify() -> Tuple[bool, str]
            Check whether this is a valid Array
    Note:
        This class is a subclass of ArrayConfig.
    """

    file_name: Path = attr.ib(
        metadata={"help_text": "The file name of the zarr container."}
    )
    dataset: str = attr.ib(
        metadata={
            "help_text": "The name of your dataset. May include '/' characters for nested heirarchies"
        }
    )
    snap_to_grid: Optional[Coordinate] = attr.ib(
        default=None,
        metadata={
            "help_text": "If you need to make sure your ROI's align with a specific voxel_size"
        },
    )
    _axes: Optional[List[str]] = attr.ib(
        default=None, metadata={"help_text": "The axis_names of your data!"}
    )
    mode: Optional[str] = attr.ib(
        default="a", metadata={"help_text": "The access mode!"}
    )

    def array(self, mode="r"):
        return open_ds(f"{self.file_name}/{self.dataset}", mode=mode)

    def verify(self) -> Tuple[bool, str]:
        """
        Check whether this is a valid Array

        Returns:
            Tuple[bool, str]: A tuple of a boolean and a string. The boolean indicates whether the Array is valid or not.
            The string provides a reason why the Array is not valid.
        Raises:
            NotImplementedError: This method is not implemented for this Array
        Examples:
            >>> zarr_array_config = ZarrArrayConfig(
            ...     file_name=Path("data.zarr"),
            ...     dataset="data",
            ...     snap_to_grid=Coordinate(1, 1, 1),
            ...     _axes=["x", "y", "z"]
            ... )
            >>> zarr_array_config.verify()
            (True, 'No validation for this Array')
        Note:
            This method is not implemented for this Array
        """
        if not self.file_name.exists():
            return False, f"{self.file_name} does not exist!"
        elif not (
            self.file_name.name.endswith(".zarr") or self.file_name.name.endswith(".n5")
        ):
            return False, f"{self.file_name} is not a zarr or n5 container"
        elif not (self.file_name / self.dataset).exists():
            return False, f"{self.dataset} is not contained in {self.file_name}"
        return True, "No validation for this Array"

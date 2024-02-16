import attr
from .array_config import ArrayConfig
from .zarr_array import ZarrArray
from funlib.geometry import Coordinate
from pathlib import Path
from typing import Optional, List, Tuple


@attr.s
class ZarrArrayConfig(ArrayConfig):
    """
    A configuration class to setup the needs for a zarr array.

    Attributes:
        array_type (ZarrArray): Type of the array for the given config.
        file_name (Path): The file name of the zarr container.
        dataset (str): The name of the dataset. You can use '/' characters for nested heirarchies.
        snap_to_grid (Optional[Coordinate]): To align the ROI's with a specific voxel_size if needed.
        _axes (Optional[List[str]]): Define the axes of data.
    """
    array_type = ZarrArray

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
        default=None, metadata={"help_text": "The axes of your data!"}
    )

    def verify(self) -> Tuple[bool, str]:
        """
        Verify the existence and validity of the array.

        Returns:
            bool: Whether the array is valid.
            str: Specific error message if the array is not valid. "No validation for this Array" if the array is valid.
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
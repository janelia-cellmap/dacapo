import attr

from .array_config import ArrayConfig
from .zarr_array import ZarrArray

from funlib.geometry import Coordinate

from upath import UPath as Path

from typing import Optional, List, Tuple


@attr.s
class ZarrArrayConfig(ArrayConfig):
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
    mode: Optional[str] = attr.ib(
        default="a", metadata={"help_text": "The access mode!"}
    )

    def verify(self) -> Tuple[bool, str]:
        if not self.file_name.exists():
            return False, f"{self.file_name} does not exist!"
        elif not (
            self.file_name.name.endswith(".zarr") or self.file_name.name.endswith(".n5")
        ):
            return False, f"{self.file_name} is not a zarr or n5 container"
        elif not (self.file_name / self.dataset).exists():
            return False, f"{self.dataset} is not contained in {self.file_name}"
        return True, "No validation for this Array"

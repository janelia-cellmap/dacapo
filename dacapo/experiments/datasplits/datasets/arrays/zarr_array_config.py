import attr

from .array_config import ArrayConfig
from .zarr_array import ZarrArray

from pathlib import Path

@attr.s
class ZarrArrayConfig(ArrayConfig):
    """This config class provides the necessary configuration for a zarr array"""

    array_type = ZarrArray

    file_name: Path = attr.ib(
        metadata={"help_text": "The file name of the zarr container."}
    )
    dataset: str = attr.ib(
        metadata={
            "help_text": "The name of your dataset. May include '/' characters for nested heirarchies"
        }
    )

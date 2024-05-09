import attr

from .array_config import ArrayConfig
from .crop_array import CropArray

from funlib.geometry import Roi


@attr.s
class CropArrayConfig(ArrayConfig):
    """
    This config class provides the necessary configuration for cropping an
    Array to a smaller ROI. Especially useful for validation volumes that may
    be too large for quick evaluation. The ROI is specified in the config. The
    cropped Array will have the same dtype as the source Array.

    Attributes:
        source_array_config (ArrayConfig): The Array to crop
        roi (Roi): The ROI for cropping
    Methods:
        from_toml(cls, toml_path: str) -> CropArrayConfig:
            Load the CropArrayConfig from a TOML file
        to_toml(self, toml_path: str) -> None:
            Save the CropArrayConfig to a TOML file
        create_array(self) -> CropArray:
            Create the CropArray from the config
    Note:
        This class is a subclass of ArrayConfig and inherits all its attributes
        and methods. The only difference is that the array_type is CropArray.
    """

    array_type = CropArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array to crop"}
    )

    roi: Roi = attr.ib(metadata={"help_text": "The ROI for cropping"})

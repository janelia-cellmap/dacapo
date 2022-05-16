import attr

from .array_config import ArrayConfig
from .crop_array import CropArray
from .array_config import ArrayConfig

from funlib.geometry import Roi


@attr.s
class CropArrayConfig(ArrayConfig):
    """This config class provides the necessary configuration for cropping an
    Array to a smaller ROI. Especially useful for validation volumes that may
    be too large for quick evaluation"""

    array_type = CropArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array to crop"}
    )

    roi: Roi = attr.ib(metadata={"help_text": "The ROI for cropping"})

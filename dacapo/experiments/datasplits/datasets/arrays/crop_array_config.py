import attr

from .array_config import ArrayConfig
from .crop_array import CropArray

from funlib.geometry import Roi


@attr.s
class CropArrayConfig(ArrayConfig):
    

    array_type = CropArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array to crop"}
    )

    roi: Roi = attr.ib(metadata={"help_text": "The ROI for cropping"})

import attr

from .array_config import ArrayConfig
from .crop_array import CropArray

from funlib.geometry import Roi


@attr.s
class CropArrayConfig(ArrayConfig):
    """
    A subclass of ArrayConfig that represents configurations for array cropping.

    This configuration class provides the necessary details for cropping an Array 
    to a smaller Region of Interest(ROI) especially useful for validation volumes 
    that might be too huge for quick evaluation

    Attributes:
        array_type (CropArray): a CropArray instance.
        source_array_config (ArrayConfig): the Array that is to be cropped.
        roi (Roi): the Region Of Interest to crop the array to.

    """

    array_type = CropArray

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array to crop"}
    )

    roi: Roi = attr.ib(metadata={"help_text": "The ROI for cropping"})

import attr

from .array_config import ArrayConfig
from funlib.persistence import Array

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

    source_array_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The Array to crop"}
    )

    roi: Roi = attr.ib(metadata={"help_text": "The ROI for cropping"})

    def array(self, mode: str = "r") -> Array:
        source_array = self.source_array_config.array(mode)
        roi_slices = getattr(source_array, "_Array__slices")(self.roi)
        out_array = Array(
            source_array.data[roi_slices],
            self.roi.offset,
            source_array.voxel_size,
            source_array.axis_names,
            source_array.units,
        )
        return out_array

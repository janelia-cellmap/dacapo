from .augment_abc import AugmentABC

import attr
import gunpowder as gp


@attr.s
class IntensityAugment(AugmentABC):
    scale_min: float = attr.ib(metadata={"help_text": "min scale to augment your data"})
    scale_max: float = attr.ib(metadata={"help_text": "max scale to augment your data"})
    shift_min: float = attr.ib(metadata={"help_text": "min shift to augment your data"})
    shift_max: float = attr.ib(metadata={"help_text": "max shift to augment your data"})
    z_section_wise: bool = attr.ib(
        default=False,
        metadata={
            "help_text": "Perform the augmentation z-section wise. Requires 3D arrays and "
            "assumes that z is the first dimension."
        },
    )
    clip: bool = attr.ib(
        default=True,
        metadata={
            "help_text": "Set to False if modified values should not be clipped to [0, 1] "
            "Disables range check!"
        },
    )

    def node(self, array):
        return gp.IntensityAugment(
            self.array,
            self.scale_min,
            self.scale_max,
            self.shift_min,
            self.shift_max,
            self.z_section_wise,
            self.clip,
        )

from .augment_config import AugmentConfig

import gunpowder as gp

import attr


@attr.s
class IntensityScaleShiftAugmentConfig(AugmentConfig):
    scale: float = attr.ib(
        metadata={"help_text": "A constant to scale your intensities."}
    )
    shift: float = attr.ib(
        metadata={"help_text": "A constant to shift your intensities."}
    )

    def node(self, raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None):
        return gp.IntensityScaleShift(raw_key, scale=self.scale, shift=self.shift)

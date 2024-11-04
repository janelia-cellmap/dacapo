from .augment_config import AugmentConfig

import gunpowder as gp

import attr

from typing import Tuple


@attr.s
class IntensityAugmentConfig(AugmentConfig):
    scale: Tuple[float, float] = attr.ib(
        metadata={"help_text": "A range within which to choose a random scale factor."}
    )
    shift: Tuple[float, float] = attr.ib(
        metadata={
            "help_text": "A range within which to choose a random additive shift."
        }
    )
    clip: bool = attr.ib(
        default=True,
        metadata={
            "help_text": "Set to False if modified values should not be clipped to [0, 1]"
        },
    )

    def node(self, raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None):
        return gp.IntensityAugment(
            raw_key,
            scale_min=self.scale[0],
            scale_max=self.scale[1],
            shift_min=self.shift[0],
            shift_max=self.shift[1],
            clip=self.clip,
        )

from .augment_config import AugmentConfig
from dacapo.gp.gamma_noise import GammaAugment

import gunpowder as gp

import attr

from typing import Tuple


@attr.s
class GammaAugmentConfig(AugmentConfig):
    

    gamma_range: Tuple[float, float] = attr.ib(
        metadata={
            "help_text": "The range (min/max) of gamma noise to apply to your data."
        }
    )

    def node(self, raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None):
        
        return GammaAugment(
            [raw_key], gamma_min=self.gamma_range[0], gamma_max=self.gamma_range[1]
        )

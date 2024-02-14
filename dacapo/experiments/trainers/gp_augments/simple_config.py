from .augment_config import AugmentConfig

import gunpowder as gp

import attr


@attr.s
class SimpleAugmentConfig(AugmentConfig):
    def node(self, _raw_key=None, _gt_key=None, _mask_key=None):
        return gp.SimpleAugment()

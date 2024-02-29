from .augment_config import AugmentConfig

import gunpowder as gp

import attr


@attr.s
class GaussianNoiseAugmentConfig(AugmentConfig):
    mean: float = attr.ib(
        metadata={"help_text": "The mean of the gaussian noise to apply to your data."},
        default=0.0,
    )
    var: float = attr.ib(
        metadata={"help_text": "The variance of the gaussian noise."},
        default=0.05,
    )

    def node(self, raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None):
        return gp.NoiseAugment(
            array=raw_key, mode="gaussian", mean=self.mean, var=self.var
        )

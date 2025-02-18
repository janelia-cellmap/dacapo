from .augment_config import AugmentConfig

import gunpowder as gp

import attr

from typing import Tuple


@attr.s
class NoiseAugmentConfig(AugmentConfig):
    """
    This class manages the configuration of gamma augmentation for a given dataset.

    Attributes:
        gamma_range: A tuple of float values represents the min and max range of gamma noise
        to apply on the raw data.
    Methods:
        node(): Constructs a node in the augmentation pipeline.
    """

    kwargs: dict[str, any] = attr.ib(
        metadata={
            "help_text": "key word arguments for skimage `random_noise`. For more details see "
            "https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise"
        },
        factory=lambda: dict(),
    )

    def node(self, raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None):
        """
        Constructs a node in the augmentation pipeline.
        """
        return gp.NoiseAugment(raw_key, **self.kwargs)

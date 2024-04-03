from .augment_config import AugmentConfig
from dacapo.gp.gamma_noise import GammaAugment

import gunpowder as gp

import attr

from typing import Tuple


@attr.s
class GammaAugmentConfig(AugmentConfig):
    """
    This class manages the configuration of gamma augmentation for a given dataset.

    Attributes:
        gamma_range: A tuple of float values represents the min and max range of gamma noise
        to apply on the raw data.
    Methods:
        node(): Constructs a node in the augmentation pipeline.
    """

    gamma_range: Tuple[float, float] = attr.ib(
        metadata={
            "help_text": "The range (min/max) of gamma noise to apply to your data."
        }
    )

    def node(self, raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None):
        """
        Constructs a node in the augmentation pipeline.

        Args:
            raw_key (gp.ArrayKey): Key to an Array (volume) in the pipeline
            _gt_key (gp.ArrayKey, optional): Ground Truth key, not used in this function. Defaults to None.
            _mask_key (gp.ArrayKey, optional): Mask Key, not used in this function. Defaults to None.
        Returns:
            GammaAugment instance: The augmentation method to be applied on the source data.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> node = gamma_augment_config.node(raw_key)
        """
        return GammaAugment(
            [raw_key], gamma_min=self.gamma_range[0], gamma_max=self.gamma_range[1]
        )

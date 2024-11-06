from .augment_config import AugmentConfig

import gunpowder as gp

import attr

from typing import Tuple


@attr.s
class IntensityAugmentConfig(AugmentConfig):
    """
    This class is an implementation of AugmentConfig that applies intensity augmentations.

    Attributes:
        scale (Tuple[float, float]): A range within which to choose a random scale factor.
        shift (Tuple[float, float]): A range within which to choose a random additive shift.
        clip (bool): Set to False if modified values should not be clipped to [0, 1]
    Methods:
        node(raw_key, _gt_key=None, _mask_key=None): Get a gp.IntensityAugment node.

    """

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
    augmentation_probability: float = attr.ib(
        default=1.0,
        metadata={"help_text": "Probability of applying the augmentation."},
    )

    def node(self, raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None):
        """
        Get a gp.IntensityAugment node.

        Args:
            raw_key (gp.ArrayKey): Key for raw data.
            _gt_key ([type], optional): Specific key for ground truth data, not used in this implementation. Defaults to None.
            _mask_key ([type], optional): Specific key for mask data, not used in this implementation. Defaults to None.
        Returns:
            gunpowder.IntensityAugment : Intensity augmentation node which can be incorporated in the pipeline.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> node = intensity_augment_config.node(raw_key)
        """
        return gp.IntensityAugment(
            raw_key,
            scale_min=self.scale[0],
            scale_max=self.scale[1],
            shift_min=self.shift[0],
            shift_max=self.shift[1],
            clip=self.clip,
            p=self.augmentation_probability,
        )

import gunpowder as gp

import attr

from abc import ABC, abstractmethod


@attr.s
class AugmentConfig(ABC):
    """
    Abstraction class for augmentation configurations in gunpowder.
    Each augmentation must have a configuration class derived from this.
    """

    @abstractmethod
    def node(
        self, raw_key: gp.ArrayKey, gt_key: gp.ArrayKey, mask_key: gp.ArrayKey
    ) -> gp.BatchFilter:
        """
        Create a gunpowder node that applies this augmentation.

        Args:
            raw_key (gp.ArrayKey): The key for the raw data array.
            gt_key (gp.ArrayKey): The key for the ground truth data array.
            mask_key (gp.ArrayKey): The key for the masking data array.

        Returns:
            gp.BatchFilter: The resulting gunpowder node that applies 
            this augmentation.
        """
        pass

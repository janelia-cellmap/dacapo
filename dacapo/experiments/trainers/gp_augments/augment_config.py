import gunpowder as gp

import attr

from abc import ABC, abstractmethod


@attr.s
class AugmentConfig(ABC):
    """
    Base class for gunpowder augment configurations. Each subclass of a `Augment`
    should have a corresponding config class derived from `AugmentConfig`.
    """

    @abstractmethod
    def node(
        self, raw_key: gp.ArrayKey, gt_key: gp.ArrayKey, mask_key: gp.ArrayKey
    ) -> gp.BatchFilter:
        """
        return a gunpowder node that performs this augmentation
        """

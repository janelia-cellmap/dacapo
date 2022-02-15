import attr

from abc import ABC, abstractmethod

@attr.s
class AugmentConfig:
    """
    Base class for gunpowder augment configurations. Each subclass of a `Augment`
    should have a corresponding config class derived from `AugmentConfig`.
    """

    @abstractmethod
    def node(self, raw_key, gt_key, mask_key):
        """
        return a gunpowder node that performs this augmentation
        """
        pass
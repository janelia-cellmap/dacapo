import gunpowder as gp

import attr

from abc import ABC, abstractmethod


@attr.s
class AugmentConfig(ABC):
    """
    Base class for gunpowder augment configurations. Each subclass of a `Augment`
    should have a corresponding config class derived from `AugmentConfig`.

    Attributes:
        _raw_key: Key for raw data. Not used in this implementation. Defaults to None.
        _gt_key: Key for ground truth data. Not used in this implementation. Defaults to None.
        _mask_key: Key for mask data. Not used in this implementation. Defaults to None.
    Methods:
        node(_raw_key=None, _gt_key=None, _mask_key=None): Get a gp.Augment node.

    """

    @abstractmethod
    def node(
        self, raw_key: gp.ArrayKey, gt_key: gp.ArrayKey, mask_key: gp.ArrayKey
    ) -> gp.BatchFilter:
        """
        Get a gunpowder augment node.

        Args:
            raw_key (gp.ArrayKey): Key for raw data.
            gt_key (gp.ArrayKey): Key for ground truth data.
            mask_key (gp.ArrayKey): Key for mask data.
        Returns:
            gunpowder.BatchFilter : Augmentation node which can be incorporated in the pipeline.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> node = augment_config.node(raw_key, gt_key, mask_key)

        """
        pass

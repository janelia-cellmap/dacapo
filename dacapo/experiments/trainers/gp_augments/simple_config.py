```python
from .augment_config import AugmentConfig

import gunpowder as gp

import attr

@attr.s
class SimpleAugmentConfig(AugmentConfig):
    """
    This class is an implementation of AugmentConfig that applies simple augmentations.
    
    Arguments:
        _raw_key: Key for raw data. Not used in this implementation. Defaults to None.
        _gt_key: Key for ground truth data. Not used in this implementation. Defaults to None.
        _mask_key: Key for mask data. Not used in this implementation. Defaults to None.

    Returns:
        Gunpowder SimpleAugment Node: A node that can be included in a pipeline to perform simple data augmentations.
    """

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None):
        """
        Get a gp.SimpleAugment node.

        Args:
            _raw_key ([type], optional): Specific key for raw data, not used in this implementation. Defaults to None.
            _gt_key ([type], optional): Specific key for ground truth data, not used in this implementation. Defaults to None.
            _mask_key ([type], optional): Specific key for mask data, not used in this implementation. Defaults to None.

        Returns:
            gunpowder.SimpleAugment : Simple augmentation node which can be incorporated in the pipeline.
        """
        return gp.SimpleAugment()
```

import gunpowder as gp

import attr

from abc import ABC, abstractmethod


@attr.s
class AugmentConfig(ABC):
    @abstractmethod
    def node(
        self, raw_key: gp.ArrayKey, gt_key: gp.ArrayKey, mask_key: gp.ArrayKey
    ) -> gp.BatchFilter:
        pass

from .augment_abc import AugmentABC

import attr
import gunpowder as gp

from typing import Optional, List


@attr.s
class SimpleAugment(AugmentABC):
    mirror_only: List[int] = attr.ib()
    transpose_only: List[int] = attr.ib()
    mirror_probs: Optional[List[float]] = attr.ib(default=None)
    transpose_probs: Optional[List[float]] = attr.ib(default=None)

    def node(self, array=None):
        return gp.SimpleAugment(
            self.mirror_only,
            self.transpose_only,
            self.mirror_probs,
            self.transpose_probs,
        )

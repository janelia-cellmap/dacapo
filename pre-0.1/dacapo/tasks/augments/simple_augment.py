from .augment_abc import AugmentABC

import attr
import gunpowder as gp

from typing import Optional, List


@attr.s
class SimpleAugment(AugmentABC):
    mirror_only: List[int] = attr.ib(
        metadata={
            "help_text": "The axes to mirror. This lets you exclude some axes "
            "if mirroring doesn't make sense."
        }
    )
    transpose_only: List[int] = attr.ib(
        metadata={
            "help_text": "The axes to transpose. This lets you exclude some axes "
            "if transposing doesn't make sense."
        }
    )
    mirror_probs: Optional[List[float]] = attr.ib(
        default=None, metadata={"help_text": "The probability of mirroring each axis."}
    )
    transpose_probs: Optional[List[float]] = attr.ib(
        default=None,
        metadata={"help_text": "The *psuedo* probability of transposing each axis."},
    )

    def node(self, array=None):
        return gp.SimpleAugment(
            self.mirror_only,
            self.transpose_only,
            self.mirror_probs,
            self.transpose_probs,
        )

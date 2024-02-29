from typing import List, Optional
from .augment_config import AugmentConfig

import gunpowder as gp

import attr


@attr.s
class SimpleAugmentConfig(AugmentConfig):
    mirror_only: Optional[List[int]] = attr.ib(
        default=None,
        metadata={
            "help_text": (
                "If set, only mirror between the given axes. This is useful to exclude channels that have a set direction, like time."
            )
        },
    )
    transpose_only: Optional[List[int]] = attr.ib(
        default=None,
        metadata={
            "help_text": (
                "If set, only transpose between the given axes. This is useful to exclude channels that have a set direction, like time."
            )
        },
    )
    mirror_probs: Optional[List[float]] = attr.ib(
        default=None,
        metadata={
            "help_text": (
                "Probability of mirroring along each axis. Defaults to 0.5 for each axis."
            )
        },
    )
    transpose_probs: Optional[List[float]] = attr.ib(
        default=None,
        metadata={
            "help_text": (
                "Probability of transposing along each axis. Defaults to 0.5 for each axis."
            )
        },
    )

    def node(
        self,
        _raw_key=None,
        _gt_key=None,
        _mask_key=None,
    ):
        return gp.SimpleAugment(
            self.mirror_only,
            self.transpose_only,
            self.mirror_probs,
            self.transpose_probs,
        )

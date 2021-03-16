import attr

from .model_abc import ModelABC

from typing import List, Optional
from enum import Enum


class ConvPaddingOption(Enum):
    VALID = "valid"
    SAME = "same"


@attr.s
class VGGNet(ModelABC):
    # standard model attributes
    input_shape: List[int] = attr.ib()
    output_shape: Optional[List[int]] = attr.ib()
    fmaps_out: int = attr.ib()

    def instantiate(self, fmaps_in: int):
        raise NotImplementedError()
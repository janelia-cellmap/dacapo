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
    input_shape: List[int] = attr.ib(metadata={"help_text": "The input shape."})
    output_shape: Optional[List[int]] = attr.ib(
        metadata={"help_text": "The output shape."}
    )
    fmaps_out: int = attr.ib(
        metadata={"help_text": "The number of featuremaps provided."}
    )

    def instantiate(self, fmaps_in: int):
        raise NotImplementedError()
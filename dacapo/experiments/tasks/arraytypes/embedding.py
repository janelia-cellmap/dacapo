from .arraytype import ArrayType

import attr

from typing import Dict


@attr.s
class EmbeddingArray(ArrayType):
    """
    A generic output of a model that could represent almost anything. Assumed to be
    float, interpolatable, and have sum number of channels.
    """

    embedding_dims: int = attr.ib(
        metadata={"help_text": "The dimension of your embedding."}
    )

    @property
    def interpolatable(self) -> bool:
        return True

from .arraytype import ArrayType

import attr


@attr.s
class EmbeddingArray(ArrayType):
    embedding_dims: int = attr.ib(
        metadata={"help_text": "The dimension of your embedding."}
    )

    @property
    def interpolatable(self) -> bool:
        return True

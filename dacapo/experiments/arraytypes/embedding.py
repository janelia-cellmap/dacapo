from .arraytype import ArrayType

import attr


@attr.s
class EmbeddingArray(ArrayType):
    """
    A generic output of a model that could represent almost anything. Assumed to be
    float, interpolatable, and have sum number of channels. The channels are not
    specified, and the array can be of any shape.

    Attributes:
        embedding_dims (int): The dimension of your embedding.
    Methods:
        interpolatable():
            It is a method that returns True.
    Note:
        This class is used to represent an EmbeddingArray object in the system.
    """

    embedding_dims: int = attr.ib(
        metadata={"help_text": "The dimension of your embedding."}
    )

    @property
    def interpolatable(self) -> bool:
        """
        Method to return True.

        Returns:
            bool
                Returns a boolean value of True representing that the values are interpolatable.
        Raises:
            NotImplementedError
                This method is not implemented in this class.
        Examples:
            >>> embedding_array = EmbeddingArray(embedding_dims=10)
            >>> embedding_array.interpolatable
            True
        Note:
            This method is used to check if the array is interpolatable.
        """
        return True

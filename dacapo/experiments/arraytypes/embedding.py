"""
A Google Style Multi-Line Docstring Format is shown below.

This module contains the Embedding array class and its attributes.

Classes:
    EmbeddingArray(ArrayType): Returns the embedding array class.
"""


@attr.s
class EmbeddingArray(ArrayType):
    """
    A class used to represent the Embedding Array. 

    ...

    Attributes
    ----------
    embedding_dims : int
        The dimension of your embedding, default is None

    Methods
    -------
    interpolatable(self) -> bool
        
    """

    embedding_dims: int = attr.ib(
        metadata={"help_text": "The dimension of your embedding."}
    )
    """
    defines the embedding dimension of your array.

    Parameters
    ----------
    metadata["help_text"] : str
        a help text which explains the role of embedding_dims.

    Raises
    ------
    None

    Returns
    -------
    None
    """

    @property
    def interpolatable(self) -> bool:
        """
        Function which returns True as per script code.
        
        Properties
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        bool
            Always returns True.
        """
        
        return True

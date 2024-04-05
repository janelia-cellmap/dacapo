import attr

from typing import List


@attr.s(frozen=True)
class PostProcessorParameters:
    """
    Base class for post-processor parameters. Post-processor parameters are
    immutable objects that define the parameters of a post-processor. The
    parameters are used to configure the post-processor.

    Attributes:
        id: The identifier of the post-processor parameter.
    Methods:
        parameter_names: Get the names of the parameters.
    Note:
        This class is immutable. Once created, the values of its attributes
        cannot be changed.

    """

    id: int = attr.ib()

    @property
    def parameter_names(self) -> List[str]:
        """
        Get the names of the parameters.

        Returns:
            A list of parameter names.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> parameters = PostProcessorParameters(0)
            >>> parameters.parameter_names
            ["id"]
        Note:
            This method must be implemented in the subclass. It should return a
            list of parameter names.
        """
        return ["id"]


# TODO: Add parameter_names to subclasses

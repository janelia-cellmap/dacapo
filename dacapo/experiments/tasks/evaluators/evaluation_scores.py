import attr

from abc import abstractmethod
from typing import Tuple, List, Union


@attr.s
class EvaluationScores:
    """
    Base class for evaluation scores. This class is used to store the evaluation scores for a task.
    The scores include the evaluation criteria. The class also provides methods to determine whether higher is better for a given criterion,
    the bounds for a given criterion, and whether to store the best score for a given criterion.

    Attributes:
        criteria : List[str]
            the evaluation criteria
    Methods:
        higher_is_better(criterion)
            Return whether higher is better for the given criterion.
        bounds(criterion)
            Return the bounds for the given criterion.
        store_best(criterion)
            Return whether to store the best score for the given criterion.
    Note:
        The EvaluationScores class is used to store the evaluation scores for a task. All evaluation scores should inherit from this class.

    """

    @property
    @abstractmethod
    def criteria(self) -> List[str]:
        """
        The evaluation criteria.

        Returns:
            List[str]
                the evaluation criteria
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluation_scores = EvaluationScores()
            >>> evaluation_scores.criteria
            ["criterion1", "criterion2"]
        Note:
            This function is used to return the evaluation criteria.
        """
        pass

    @staticmethod
    @abstractmethod
    def higher_is_better(criterion: str) -> bool:
        """
        Wether or not higher is better for this criterion.

        Args:
            criterion : str
                the evaluation criterion
        Returns:
            bool
                whether higher is better for this criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluation_scores = EvaluationScores()
            >>> criterion = "criterion1"
            >>> evaluation_scores.higher_is_better(criterion)
            True
        Note:
            This function is used to determine whether higher is better for a given criterion.

        """
        pass

    @staticmethod
    @abstractmethod
    def bounds(
        criterion: str,
    ) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """
        The bounds for this criterion.

        Args:
            criterion : str
                the evaluation criterion
        Returns:
            Tuple[Union[int, float, None], Union[int, float, None]]
                the bounds for this criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluation_scores = EvaluationScores()
            >>> criterion = "criterion1"
            >>> evaluation_scores.bounds(criterion)
            (0, 1)
        Note:
            This function is used to return the bounds for the given criterion.

        """
        pass

    @staticmethod
    @abstractmethod
    def store_best(criterion: str) -> bool:
        """
        Whether or not to save the best validation block and model
        weights for this criterion.

        Args:
            criterion : str
                the evaluation criterion
        Returns:
            bool
                whether to store the best score for this criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluation_scores = EvaluationScores()
            >>> criterion = "criterion1"
            >>> evaluation_scores.store_best(criterion)
            True
        Note:
            This function is used to return whether to store the best score for the given criterion.
        """
        pass

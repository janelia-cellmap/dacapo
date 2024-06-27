from .evaluation_scores import EvaluationScores
import attr

from typing import Tuple, Union


@attr.s
class DummyEvaluationScores(EvaluationScores):
    """
    The evaluation scores for the dummy task. The scores include the frizz level and blipp score. A higher frizz level indicates more frizz, while a higher blipp score indicates better performance.

    Attributes:
        frizz_level : float
            the frizz level
        blipp_score : float
            the blipp score
    Methods:
        higher_is_better(criterion)
            Return whether higher is better for the given criterion.
        bounds(criterion)
            Return the bounds for the given criterion.
        store_best(criterion)
            Return whether to store the best score for the given criterion.
    Note:
        The DummyEvaluationScores class is used to store the evaluation scores for the dummy task. The class also provides methods to determine whether higher is better for a given criterion, the bounds for a given criterion, and whether to store the best score for a given criterion.
    """

    criteria = ["frizz_level", "blipp_score"]

    frizz_level: float = attr.ib(default=float("nan"))
    blipp_score: float = attr.ib(default=float("nan"))

    @staticmethod
    def higher_is_better(criterion: str) -> bool:
        """
        Return whether higher is better for the given criterion.

        Args:
            criterion : str
                the evaluation criterion
        Returns:
            bool
                whether higher is better for this criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> DummyEvaluationScores.higher_is_better("frizz_level")
            True
        Note:
            This function is used to determine whether higher is better for the given criterion.
        """
        mapping = {
            "frizz_level": True,
            "blipp_score": False,
        }
        return mapping[criterion]

    @staticmethod
    def bounds(
        criterion: str,
    ) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """
        Return the bounds for the given criterion.

        Args:
            criterion : str
                the evaluation criterion
        Returns:
            Tuple[Union[int, float, None], Union[int, float, None]]
                the bounds for the given criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> DummyEvaluationScores.bounds("frizz_level")
            (0.0, 1.0)
        Note:
            This function is used to return the bounds for the given criterion.
        """
        mapping = {
            "frizz_level": (0.0, 1.0),
            "blipp_score": (0.0, 1.0),
        }
        return mapping[criterion]

    @staticmethod
    def store_best(criterion: str) -> bool:
        """
        Return whether to store the best score for the given criterion.

        Args:
            criterion : str
                the evaluation criterion
        Returns:
            bool
                whether to store the best score for the given criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> DummyEvaluationScores.store_best("frizz_level")
            True
        Note:
            This function is used to determine whether to store the best score for the given criterion.
        """
        return True

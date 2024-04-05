from .evaluation_scores import EvaluationScores
import attr

from typing import Tuple, Union


@attr.s
class InstanceEvaluationScores(EvaluationScores):
    """
    The evaluation scores for the instance segmentation task. The scores include the variation of information (VOI) split, VOI merge, and VOI.

    Attributes:
        voi_split : float
            the variation of information (VOI) split
        voi_merge : float
            the variation of information (VOI) merge
        voi : float
            the variation of information (VOI)
    Methods:
        higher_is_better(criterion)
            Return whether higher is better for the given criterion.
        bounds(criterion)
            Return the bounds for the given criterion.
        store_best(criterion)
            Return whether to store the best score for the given criterion.
    Note:
        The InstanceEvaluationScores class is used to store the evaluation scores for the instance segmentation task.
    """

    criteria = ["voi_split", "voi_merge", "voi"]

    voi_split: float = attr.ib(default=float("nan"))
    voi_merge: float = attr.ib(default=float("nan"))

    @property
    def voi(self):
        """
        Return the average of the VOI split and VOI merge.

        Returns:
            float
                the average of the VOI split and VOI merge
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> instance_evaluation_scores = InstanceEvaluationScores(voi_split=0.1, voi_merge=0.2)
            >>> instance_evaluation_scores.voi
            0.15
        Note:
            This function is used to calculate the average of the VOI split and VOI merge.
        """
        return (self.voi_split + self.voi_merge) / 2

    @staticmethod
    def higher_is_better(criterion: str) -> bool:
        """
        Return whether higher is better for the given criterion.

        Args:
            criterion : str
                the evaluation criterion
        Returns:
            bool
                whether higher is better for the given criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> InstanceEvaluationScores.higher_is_better("voi_split")
            False
        Note:
            This function is used to determine whether higher is better for the given criterion.
        """
        mapping = {
            "voi_split": False,
            "voi_merge": False,
            "voi": False,
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
            >>> InstanceEvaluationScores.bounds("voi_split")
            (0, 1)
        Note:
            This function is used to return the bounds for the given criterion.

        """
        mapping = {
            "voi_split": (0, 1),
            "voi_merge": (0, 1),
            "voi": (0, 1),
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
            >>> InstanceEvaluationScores.store_best("voi_split")
            True
        Note:
            This function is used to determine whether to store the best score for the given criterion.
        """
        return True

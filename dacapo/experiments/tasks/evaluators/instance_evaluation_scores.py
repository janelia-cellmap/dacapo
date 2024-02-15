from .evaluation_scores import EvaluationScores
import attr

from typing import Tuple


@attr.s
class InstanceEvaluationScores(EvaluationScores):
    """
    InstanceEvaluationScores is for storing and computing VOI (Variation of Information) related evaluation 
    scores for instance segmentation tasks. It handles VOI split and merge scores and 
    provides utility methods for score analysis and comparison.

    Attributes:
        voi_split (float): Score for the VOI split metric.
        voi_merge (float): Score for the VOI merge metric.
    """
    criteria = ["voi_split", "voi_merge", "voi"]

    voi_split: float = attr.ib(default=float("nan"))
    voi_merge: float = attr.ib(default=float("nan"))

    @property
    def voi(self):
        """
        Calculates the average of VOI split and VOI merge scores.

        Returns:
            float: The average VOI score.
        """
        return (self.voi_split + self.voi_merge) / 2

    @staticmethod
    def higher_is_better(criterion: str) -> bool:
        """
        Determines if a higher score is better for a given criterion.

        Args:
            criterion (str): The evaluation criterion.

        Returns:
            bool: False for all criteria in this class, indicating that a lower score is better.
        """
        mapping = {
            "voi_split": False,
            "voi_merge": False,
            "voi": False,
        }
        return mapping[criterion]

    @staticmethod
    def bounds(criterion: str) -> Tuple[float, float]:
        """
        Provides the bounds for the possible values of a given criterion.

        Args:
            criterion (str): The evaluation criterion.

        Returns:
            Tuple[float, float]: The lower and upper bounds for the criterion's score.
                                  For VOI-based criteria, the bounds are (0, 1).
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
        Indicates whether the best score should be stored for a given criterion.

        Args:
            criterion (str): The evaluation criterion.

        Returns:
            bool: True for all criteria in this class, indicating that the best score should be stored.
        """
        return True

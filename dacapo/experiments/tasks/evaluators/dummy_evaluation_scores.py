"""
This module provides a dummy class `DummyEvaluationScores` inherited from `EvaluationScores`, 
for testing or example purposes.
"""

from .evaluation_scores import EvaluationScores
import attr

from typing import Tuple


@attr.s
"""
A class to represent a DummyEvaluationScores.

Attributes
----------
criteria : list
    A list of predefined criteria of evaluation.

frizz_level : float
    A score for "frizz_level" criterion. The higher, the better.

blipp_score : float
    A score for "blipp_score" criterion. The lower, the better.
"""
class DummyEvaluationScores(EvaluationScores):
    criteria = ["frizz_level", "blipp_score"]

    frizz_level: float = attr.ib(default=float("nan"))
    blipp_score: float = attr.ib(default=float("nan"))

    @staticmethod
    """
    Method to return whether a higher criterion score is better.

    Parameters
    ----------
    criterion : str
        Criterion name.

    Returns
    -------
    bool
        Returns True for "frizz_level" and False for "blipp_score".
    """
    def higher_is_better(criterion: str) -> bool:
        mapping = {
            "frizz_level": True,
            "blipp_score": False,
        }
        return mapping[criterion]

    @staticmethod
    """
    Method to return the bounds of criterion score.

    Parameters
    ----------
    criterion : str
       Criterion name.

    Returns
    -------
    tuple
        Returns a tuple of lower and upper bounds for each criterion.
    """
    def bounds(criterion: str) -> Tuple[float, float]:
        mapping = {
            "frizz_level": (0.0, 1.0),
            "blipp_score": (0.0, 1.0),
        }
        return mapping[criterion]

    @staticmethod
    """
    Method to determine if the best criterion score should be stored.
    
    Parameters
    ----------
    criterion : str
        Criterion name.

    Returns
    -------
    bool
        Always returns True in this case.
    """
    def store_best(criterion: str) -> bool:
        return True

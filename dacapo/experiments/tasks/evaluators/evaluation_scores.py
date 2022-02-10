import attr

from abc import abstractmethod
from typing import Tuple

@attr.s
class EvaluationScores:
    """Base class for evaluation scores."""
    pass

    @staticmethod
    @abstractmethod
    def higher_is_better(criterion: str) -> bool:
        """
        Wether or not higher is better for this criterion.
        """
        pass

    @staticmethod
    @abstractmethod
    def bounds(criterion: str) -> Tuple[float, float]:
        """
        The bounds for this criterion
        """
        pass

    @staticmethod
    @abstractmethod
    def store_best(criterion: str) -> bool:
        """
        Whether or not to save the best validation block and model
        weights for this criterion.
        """
        pass
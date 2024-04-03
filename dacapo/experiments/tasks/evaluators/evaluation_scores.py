import attr

from abc import abstractmethod
from typing import Tuple, List, Union


@attr.s
class EvaluationScores:
    """Base class for evaluation scores."""

    @property
    @abstractmethod
    def criteria(self) -> List[str]:
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
    def bounds(
        criterion: str,
    ) -> Tuple[Union[int, float, None], Union[int, float, None]]:
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

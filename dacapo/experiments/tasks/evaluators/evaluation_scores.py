import attr

from abc import abstractmethod
from typing import Tuple, List, Union


@attr.s
class EvaluationScores:
    @property
    @abstractmethod
    def criteria(self) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def higher_is_better(criterion: str) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def bounds(
        criterion: str,
    ) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        pass

    @staticmethod
    @abstractmethod
    def store_best(criterion: str) -> bool:
        pass

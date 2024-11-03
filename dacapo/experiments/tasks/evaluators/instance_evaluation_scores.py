from .evaluation_scores import EvaluationScores
import attr

from typing import Tuple, Union


@attr.s
class InstanceEvaluationScores(EvaluationScores):
    

    criteria = ["voi_split", "voi_merge", "voi"]

    voi_split: float = attr.ib(default=float("nan"))
    voi_merge: float = attr.ib(default=float("nan"))

    @property
    def voi(self):
        
        return (self.voi_split + self.voi_merge) / 2

    @staticmethod
    def higher_is_better(criterion: str) -> bool:
        
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
        
        mapping = {
            "voi_split": (0, 1),
            "voi_merge": (0, 1),
            "voi": (0, 1),
        }
        return mapping[criterion]

    @staticmethod
    def store_best(criterion: str) -> bool:
        
        return True

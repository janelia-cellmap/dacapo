from .evaluation_scores import EvaluationScores
import attr

from typing import Tuple

@attr.s
class DummyEvaluationScores(EvaluationScores):
    criteria = ["frizz_level", "blipp_score"]

    frizz_level: float = attr.ib(default=float("nan"))
    blipp_score: float = attr.ib(default=float("nan"))

    @staticmethod
    def higher_is_better(criterion: str) -> bool:
        mapping = {
            "frizz_level": True,
            "blipp_score": False,
        }
        return mapping[criterion]

    @staticmethod
    def bounds(criterion: str) -> Tuple[float, float]:
        mapping = {
            "frizz_level": (0.0, 1.0),
            "blipp_score": (0.0, 1.0),
        }
        return mapping[criterion]

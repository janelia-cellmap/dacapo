from .evaluation_scores import EvaluationScores
import attr


@attr.s
class DummyEvaluationScores(EvaluationScores):
    criteria = ["frizz_level", "blipp_score"]

    frizz_level: float = attr.ib(default=float("nan"))
    blipp_score: float = attr.ib(default=float("nan"))

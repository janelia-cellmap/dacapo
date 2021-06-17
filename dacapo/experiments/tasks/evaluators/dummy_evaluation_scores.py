from .evaluation_scores import EvaluationScores
import attr


@attr.s
class DummyEvaluationScores(EvaluationScores):

    frizz_level: float = attr.ib()
    blipp_score: float = attr.ib()

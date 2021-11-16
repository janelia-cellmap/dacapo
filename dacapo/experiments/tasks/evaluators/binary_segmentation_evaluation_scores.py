from .evaluation_scores import EvaluationScores
import attr


@attr.s
class BinarySegmentationEvaluationScores(EvaluationScores):

    frizz_level: float = attr.ib()

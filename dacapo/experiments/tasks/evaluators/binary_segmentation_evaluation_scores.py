from .evaluation_scores import EvaluationScores
import attr


@attr.s
class BinarySegmentationEvaluationScores(EvaluationScores):
    dice: float = attr.ib()
    jaccard: float = attr.ib()
    hausdorff: float = attr.ib()
    false_negative_rate: float = attr.ib()
    false_negative_rate_with_tolerance: float = attr.ib()
    false_positive_rate: float = attr.ib()
    false_discovery_rate: float = attr.ib()
    false_positive_rate_with_tolerance: float = attr.ib()
    voi: float = attr.ib()
    mean_false_distance: float = attr.ib()
    mean_false_negative_distance: float = attr.ib()
    mean_false_positive_distance: float = attr.ib()
    mean_false_distance_clipped: float = attr.ib()
    mean_false_negative_distance_clipped: float = attr.ib()
    mean_false_positive_distance_clipped: float = attr.ib()
    precision_with_tolerance: float = attr.ib()
    recall_with_tolerance: float = attr.ib()
    f1_score_with_tolerance: float = attr.ib()
    precision: float = attr.ib()
    recall: float = attr.ib()
    f1_score: float = attr.ib()

from .evaluation_scores import EvaluationScores
import attr

from typing import Dict

maximize = {True: float("-inf"), False: float("inf")}


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

    criteria = [
        "dice",
        "jaccard",
        "hausdorff",
        "false_negative_rate",
        "false_negative_rate_with_tolerance",
        "false_positive_rate",
        "false_discovery_rate",
        "false_positive_rate_with_tolerance",
        "voi",
        "mean_false_distance",
        "mean_false_negative_distance",
        "mean_false_positive_distance",
        "mean_false_distance_clipped",
        "mean_false_negative_distance_clipped",
        "mean_false_positive_distance_clipped",
        "precision_with_tolerance",
        "recall_with_tolerance",
        "f1_score_with_tolerance",
        "precision",
        "recall",
        "f1_score",
    ]


@attr.s
class MultiChannelBinarySegmentationEvaluationScores(EvaluationScores):
    channel_scores: Dict[str, BinarySegmentationEvaluationScores] = attr.ib()

    def __attrs_post_init__(self):
        for channel, scores in self.channel_scores.items():
            for criteria in BinarySegmentationEvaluationScores.criteria:
                setattr(self, f"{channel}__{criteria}", getattr(scores, criteria))

    @property
    def criteria(self):
        return [
            f"{channel}__{criteria}"
            for channel in self.channel_scores
            for criteria in BinarySegmentationEvaluationScores.criteria
        ]

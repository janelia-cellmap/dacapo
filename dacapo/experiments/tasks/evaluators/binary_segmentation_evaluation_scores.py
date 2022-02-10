from .evaluation_scores import EvaluationScores
import attr

from typing import List, Tuple


@attr.s
class BinarySegmentationEvaluationScores(EvaluationScores):
    dice: float = attr.ib(default=float("nan"))
    jaccard: float = attr.ib(default=float("nan"))
    hausdorff: float = attr.ib(default=float("nan"))
    false_negative_rate: float = attr.ib(default=float("nan"))
    false_negative_rate_with_tolerance: float = attr.ib(default=float("nan"))
    false_positive_rate: float = attr.ib(default=float("nan"))
    false_discovery_rate: float = attr.ib(default=float("nan"))
    false_positive_rate_with_tolerance: float = attr.ib(default=float("nan"))
    voi: float = attr.ib(default=float("nan"))
    mean_false_distance: float = attr.ib(default=float("nan"))
    mean_false_negative_distance: float = attr.ib(default=float("nan"))
    mean_false_positive_distance: float = attr.ib(default=float("nan"))
    mean_false_distance_clipped: float = attr.ib(default=float("nan"))
    mean_false_negative_distance_clipped: float = attr.ib(default=float("nan"))
    mean_false_positive_distance_clipped: float = attr.ib(default=float("nan"))
    precision_with_tolerance: float = attr.ib(default=float("nan"))
    recall_with_tolerance: float = attr.ib(default=float("nan"))
    f1_score_with_tolerance: float = attr.ib(default=float("nan"))
    precision: float = attr.ib(default=float("nan"))
    recall: float = attr.ib(default=float("nan"))
    f1_score: float = attr.ib(default=float("nan"))

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

    @staticmethod
    def store_best(criterion: str) -> bool:
        # Whether or not to store the best weights/validation blocks for this
        # criterion.
        mapping = {
            "dice": False,
            "jaccard": True,
            "hausdorff": False,
            "false_negative_rate": False,
            "false_negative_rate_with_tolerance": False,
            "false_positive_rate": False,
            "false_discovery_rate": False,
            "false_positive_rate_with_tolerance": False,
            "voi": True,
            "mean_false_distance": False,
            "mean_false_positive_distance": False,
            "mean_false_negative_distance": False,
            "mean_false_distance_clipped": False,
            "mean_false_negative_distance_clipped": False,
            "mean_false_positive_distance_clipped": False,
            "precision_with_tolerance": False,
            "recall_with_tolerance": False,
            "f1_score_with_tolerance": False,
            "precision": False,
            "recall": False,
            "f1_score": True,
        }
        return mapping[criterion]


    @staticmethod
    def higher_is_better(criterion: str) -> bool:
        mapping = {
            "dice": True,
            "jaccard": True,
            "hausdorff": False,
            "false_negative_rate": False,
            "false_negative_rate_with_tolerance": False,
            "false_positive_rate": False,
            "false_discovery_rate": False,
            "false_positive_rate_with_tolerance": False,
            "voi": False,
            "mean_false_distance": False,
            "mean_false_positive_distance": False,
            "mean_false_negative_distance": False,
            "mean_false_distance_clipped": False,
            "mean_false_negative_distance_clipped": False,
            "mean_false_positive_distance_clipped": False,
            "precision_with_tolerance": True,
            "recall_with_tolerance": True,
            "f1_score_with_tolerance": True,
            "precision": True,
            "recall": True,
            "f1_score": True,
        }
        return mapping[criterion]

    @staticmethod
    def bounds(criterion: str) -> Tuple[float, float]:
        mapping = {
            "dice": (0, 1),
            "jaccard": (0, 1),
            "hausdorff": (0, float("nan")),
            "false_negative_rate": (0, 1),
            "false_negative_rate_with_tolerance": (0, 1),
            "false_positive_rate": (0, 1),
            "false_discovery_rate": (0, 1),
            "false_positive_rate_with_tolerance": (0, 1),
            "voi": (0, 1),
            "mean_false_distance": (0, float("nan")),
            "mean_false_positive_distance": (0, float("nan")),
            "mean_false_negative_distance": (0, float("nan")),
            "mean_false_distance_clipped": (0, float("nan")),
            "mean_false_negative_distance_clipped": (0, float("nan")),
            "mean_false_positive_distance_clipped": (0, float("nan")),
            "precision_with_tolerance": (0, 1),
            "recall_with_tolerance": (0, 1),
            "f1_score_with_tolerance": (0, 1),
            "precision": (0, 1),
            "recall": (0, 1),
            "f1_score": (0, 1),
        }
        return mapping[criterion]


@attr.s
class MultiChannelBinarySegmentationEvaluationScores(EvaluationScores):
    channel_scores: List[Tuple[str, BinarySegmentationEvaluationScores]] = attr.ib()

    def __attrs_post_init__(self):
        for channel, scores in self.channel_scores:
            for criteria in BinarySegmentationEvaluationScores.criteria:
                setattr(self, f"{channel}__{criteria}", getattr(scores, criteria))

    @property
    def criteria(self):
        return [
            f"{channel}__{criteria}"
            for channel, _ in self.channel_scores
            for criteria in BinarySegmentationEvaluationScores.criteria
        ]

    @staticmethod
    def higher_is_better(criterion: str) -> bool:
        _, criterion = criterion.split("__")
        return BinarySegmentationEvaluationScores.higher_is_better(criterion)

    @staticmethod
    def store_best(criterion: str) -> bool:
        _, criterion = criterion.split("__")
        return BinarySegmentationEvaluationScores.store_best(criterion)

    @staticmethod
    def bounds(criterion: str) -> Tuple[float, float]:
        _, criterion = criterion.split("__")
        return BinarySegmentationEvaluationScores.bounds(criterion)

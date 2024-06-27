from .evaluation_scores import EvaluationScores
import attr

from typing import List, Tuple, Union


@attr.s
class BinarySegmentationEvaluationScores(EvaluationScores):
    """
    Class representing evaluation scores for binary segmentation tasks.

    The metrics include:
    - Dice coefficient: 2 * |A ∩ B| / |A| + |B| ; where A and B are the binary segmentations
    - Jaccard coefficient: |A ∩ B| / |A ∪ B| ; where A and B are the binary segmentations
    - Hausdorff distance: max(h(A, B), h(B, A)) ; where h(A, B) is the Hausdorff distance between A and B
    - False negative rate: |A - B| / |A| ; where A and B are the binary segmentations
    - False positive rate: |B - A| / |B| ; where A and B are the binary segmentations
    - False discovery rate: |B - A| / |A| ; where A and B are the binary segmentations
    - VOI: Variation of Information; split and merge errors combined into a single measure of segmentation quality
    - Mean false distance: 0.5 * (mean false positive distance + mean false negative distance)
    - Mean false negative distance: mean distance of false negatives
    - Mean false positive distance: mean distance of false positives
    - Mean false distance clipped: 0.5 * (mean false positive distance clipped + mean false negative distance clipped) ; clipped to a maximum distance
    - Mean false negative distance clipped: mean distance of false negatives clipped ; clipped to a maximum distance
    - Mean false positive distance clipped: mean distance of false positives clipped ; clipped to a maximum distance
    - Precision with tolerance: TP / (TP + FP) ; where TP and FP are the true and false positives within a tolerance distance
    - Recall with tolerance: TP / (TP + FN) ; where TP and FN are the true and false positives within a tolerance distance
    - F1 score with tolerance: 2 * (Recall * Precision) / (Recall + Precision) ; where Recall and Precision are the true and false positives within a tolerance distance
    - Precision: TP / (TP + FP) ; where TP and FP are the true and false positives
    - Recall: TP / (TP + FN) ; where TP and FN are the true and false positives
    - F1 score: 2 * (Recall * Precision) / (Recall + Precision) ; where Recall and Precision are the true and false positives

    Attributes:
        dice (float): The Dice coefficient.
        jaccard (float): The Jaccard index.
        hausdorff (float): The Hausdorff distance.
        false_negative_rate (float): The false negative rate.
        false_negative_rate_with_tolerance (float): The false negative rate with tolerance.
        false_positive_rate (float): The false positive rate.
        false_discovery_rate (float): The false discovery rate.
        false_positive_rate_with_tolerance (float): The false positive rate with tolerance.
        voi (float): The variation of information.
        mean_false_distance (float): The mean false distance.
        mean_false_negative_distance (float): The mean false negative distance.
        mean_false_positive_distance (float): The mean false positive distance.
        mean_false_distance_clipped (float): The mean false distance clipped.
        mean_false_negative_distance_clipped (float): The mean false negative distance clipped.
        mean_false_positive_distance_clipped (float): The mean false positive distance clipped.
        precision_with_tolerance (float): The precision with tolerance.
        recall_with_tolerance (float): The recall with tolerance.
        f1_score_with_tolerance (float): The F1 score with tolerance.
        precision (float): The precision.
        recall (float): The recall.
        f1_score (float): The F1 score.
    Methods:
        store_best(criterion: str) -> bool: Whether or not to store the best weights/validation blocks for this criterion.
        higher_is_better(criterion: str) -> bool: Determines whether a higher value is better for a given criterion.
        bounds(criterion: str) -> Tuple[Union[int, float, None], Union[int, float, None]]: Determines the bounds for a given criterion.
    Notes:
        The evaluation scores are stored as attributes of the class. The class also contains methods to determine whether a higher value is better for a given criterion, whether or not to store the best weights/validation blocks for a given criterion, and the bounds for a given criterion.
    """

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
        """
        Determines whether or not to store the best weights/validation blocks for a given criterion.

        Args:
            criterion (str): The evaluation criterion.
        Returns:
            bool: True if the best weights/validation blocks should be stored, False otherwise.
        Raises:
            ValueError: If the criterion is not recognized.
        Examples:
            >>> BinarySegmentationEvaluationScores.store_best("dice")
            False
            >>> BinarySegmentationEvaluationScores.store_best("f1_score")
            True
        Notes:
            The method returns True if the criterion is recognized and False otherwise. Whether or not to store the best weights/validation blocks for a given criterion is determined by the mapping dictionary.

        """
        # Whether or not to store the best weights/validation blocks for this
        # criterion.
        mapping = {
            "dice": False,
            "jaccard": False,
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
        """
        Determines whether a higher value is better for a given criterion.

        Args:
            criterion (str): The evaluation criterion.
        Returns:
            bool: True if a higher value is better, False otherwise.
        Raises:
            ValueError: If the criterion is not recognized.
        Examples:
            >>> BinarySegmentationEvaluationScores.higher_is_better("dice")
            True
            >>> BinarySegmentationEvaluationScores.higher_is_better("f1_score")
            True
        Notes:
            The method returns True if the criterion is recognized and False otherwise. Whether a higher value is better for a given criterion is determined by the mapping dictionary.
        """
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
    def bounds(
        criterion: str,
    ) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """
        Determines the bounds for a given criterion. The bounds are used to determine the best value for a given criterion.

        Args:
            criterion (str): The evaluation criterion.
        Returns:
            Tuple[Union[int, float, None], Union[int, float, None]]: The lower and upper bounds for the criterion.
        Raises:
            ValueError: If the criterion is not recognized.
        Examples:
            >>> BinarySegmentationEvaluationScores.bounds("dice")
            (0, 1)
            >>> BinarySegmentationEvaluationScores.bounds("hausdorff")
            (0, nan)
        Notes:
            The method returns the lower and upper bounds for the criterion. The bounds are determined by the mapping dictionary.
        """
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
    """
    Class representing evaluation scores for multi-channel binary segmentation tasks.

    Attributes:
        channel_scores (List[Tuple[str, BinarySegmentationEvaluationScores]]): The list of channel scores.
    Methods:
        higher_is_better(criterion: str) -> bool: Determines whether a higher value is better for a given criterion.
        store_best(criterion: str) -> bool: Whether or not to store the best weights/validation blocks for this criterion.
        bounds(criterion: str) -> Tuple[Union[int, float, None], Union[int, float, None]]: Determines the bounds for a given criterion.
    Notes:
        The evaluation scores are stored as attributes of the class. The class also contains methods to determine whether a higher value is better for a given criterion, whether or not to store the best weights/validation blocks for a given criterion, and the bounds for a given criterion.
    """

    channel_scores: List[Tuple[str, BinarySegmentationEvaluationScores]] = attr.ib()

    def __attrs_post_init__(self):
        """
        Post-initialization method to set attributes for each channel and criterion.

        Raises:
            ValueError: If the criterion is not recognized.
        Examples:
            >>> channel_scores = [("channel1", BinarySegmentationEvaluationScores()), ("channel2", BinarySegmentationEvaluationScores())]
            >>> MultiChannelBinarySegmentationEvaluationScores(channel_scores)
        Notes:
            The method sets attributes for each channel and criterion. The attributes are stored as attributes of the class.
        """
        for channel, scores in self.channel_scores:
            for criteria in BinarySegmentationEvaluationScores.criteria:
                setattr(self, f"{channel}__{criteria}", getattr(scores, criteria))

    @property
    def criteria(self):
        """
        Returns a list of all criteria for all channels.

        Returns:
            List[str]: The list of criteria.
        Raises:
            ValueError: If the criterion is not recognized.
        Examples:
            >>> channel_scores = [("channel1", BinarySegmentationEvaluationScores()), ("channel2", BinarySegmentationEvaluationScores())]
            >>> MultiChannelBinarySegmentationEvaluationScores(channel_scores).criteria
        Notes:
            The method returns a list of all criteria for all channels. The criteria are stored as attributes of the class.
        """

        return [
            f"{channel}__{criteria}"
            for channel, _ in self.channel_scores
            for criteria in BinarySegmentationEvaluationScores.criteria
        ]

    @staticmethod
    def higher_is_better(criterion: str) -> bool:
        """
        Determines whether a higher value is better for a given criterion.

        Args:
            criterion (str): The evaluation criterion.
        Returns:
            bool: True if a higher value is better, False otherwise.
        Raises:
            ValueError: If the criterion is not recognized.
        Examples:
            >>> MultiChannelBinarySegmentationEvaluationScores.higher_is_better("channel1__dice")
            True
            >>> MultiChannelBinarySegmentationEvaluationScores.higher_is_better("channel1__f1_score")
            True
        Notes:
            The method returns True if the criterion is recognized and False otherwise. Whether a higher value is better for a given criterion is determined by the mapping dictionary.
        """
        _, criterion = criterion.split("__")
        return BinarySegmentationEvaluationScores.higher_is_better(criterion)

    @staticmethod
    def store_best(criterion: str) -> bool:
        """
        Determines whether or not to store the best weights/validation blocks for a given criterion.

        Args:
            criterion (str): The evaluation criterion.
        Returns:
            bool: True if the best weights/validation blocks should be stored, False otherwise.
        Raises:
            ValueError: If the criterion is not recognized.
        Examples:
            >>> MultiChannelBinarySegmentationEvaluationScores.store_best("channel1__dice")
            False
            >>> MultiChannelBinarySegmentationEvaluationScores.store_best("channel1__f1_score")
            True
        Notes:
            The method returns True if the criterion is recognized and False otherwise. Whether or not to store the best weights/validation blocks for a given criterion is determined by the mapping dictionary.
        """
        _, criterion = criterion.split("__")
        return BinarySegmentationEvaluationScores.store_best(criterion)

    @staticmethod
    def bounds(
        criterion: str,
    ) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """
        Determines the bounds for a given criterion. The bounds are used to determine the best value for a given criterion.

        Args:
            criterion (str): The evaluation criterion.
        Returns:
            Tuple[Union[int, float, None], Union[int, float, None]]: The lower and upper bounds for the criterion.
        Raises:
            ValueError: If the criterion is not recognized.
        Examples:
            >>> MultiChannelBinarySegmentationEvaluationScores.bounds("channel1__dice")
            (0, 1)
            >>> MultiChannelBinarySegmentationEvaluationScores.bounds("channel1__hausdorff")
            (0, nan)
        Notes:
            The method returns the lower and upper bounds for the criterion. The bounds are determined by the mapping dictionary.
        """
        _, criterion = criterion.split("__")
        return BinarySegmentationEvaluationScores.bounds(criterion)

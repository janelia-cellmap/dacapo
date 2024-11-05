from dacapo.utils.voi import voi
from .evaluator import Evaluator
from .binary_segmentation_evaluation_scores import (
    BinarySegmentationEvaluationScores,
    MultiChannelBinarySegmentationEvaluationScores,
)

from dacapo.tmp import open_from_identifier


import numpy as np
import SimpleITK as sitk
import lazy_property
import scipy

import itertools
import logging
from typing import List

logger = logging.getLogger(__name__)

BG = 0


class BinarySegmentationEvaluator(Evaluator):
    """
    Given a binary segmentation, compute various metrics to determine their similarity. The metrics include:
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
        clip_distance : float
            the clip distance
        tol_distance : float
            the tolerance distance
        channels : List[str]
            the channels
        criteria : List[str]
            the evaluation criteria
    Methods:
        evaluate(output_array_identifier, evaluation_array)
            Evaluate the output array against the evaluation array.
        score
            Return the evaluation scores.
    Note:
        The BinarySegmentationEvaluator class is used to evaluate the performance of a binary segmentation task.
        The class provides methods to evaluate the output array against the evaluation array and return the evaluation scores.
        All evaluation scores should inherit from this class.

        Clip distance is the maximum distance between the ground truth and the predicted segmentation for a pixel to be considered a false positive.
        Tolerance distance is the maximum distance between the ground truth and the predicted segmentation for a pixel to be considered a true positive.
        Channels are the channels of the binary segmentation.
        Criteria are the evaluation criteria.

    """

    criteria = ["jaccard", "voi"]

    def __init__(self, clip_distance: float, tol_distance: float, channels: List[str]):
        """
        Initialize the binary segmentation evaluator.

        Args:
            clip_distance : float
                the clip distance
            tol_distance : float
                the tolerance distance
            channels : List[str]
                the channels
        Raises:
            ValueError: if the clip distance is not valid
        Examples:
            >>> binary_segmentation_evaluator = BinarySegmentationEvaluator(clip_distance=200, tol_distance=40, channels=["channel1", "channel2"])
        Note:
            This function is used to initialize the binary segmentation evaluator.

            Clip distance is the maximum distance between the ground truth and the predicted segmentation for a pixel to be considered a false positive.
            Tolerance distance is the maximum distance between the ground truth and the predicted segmentation for a pixel to be considered a true positive.
            Channels are the channels of the binary segmentation.
            Criteria are the evaluation criteria.
        """
        self.clip_distance = clip_distance
        self.tol_distance = tol_distance
        self.channels = channels
        self.criteria = [
            f"{channel}__{criteria}"
            for channel, criteria in itertools.product(channels, self.criteria)
        ]

    def evaluate(self, output_array_identifier, evaluation_array):
        """
        Evaluate the output array against the evaluation array.

        Args:
            output_array_identifier : str
                the identifier of the output array
            evaluation_array : Zarr Array
                the evaluation array
        Returns:
            BinarySegmentationEvaluationScores or MultiChannelBinarySegmentationEvaluationScores
                the evaluation scores
        Raises:
            ValueError: if the output array identifier is not valid
        Examples:
            >>> binary_segmentation_evaluator = BinarySegmentationEvaluator(clip_distance=200, tol_distance=40, channels=["channel1", "channel2"])
            >>> output_array_identifier = "output_array"
            >>> evaluation_array = open_from_identifier("evaluation_array")
            >>> binary_segmentation_evaluator.evaluate(output_array_identifier, evaluation_array)
            BinarySegmentationEvaluationScores(dice=0.0, jaccard=0.0, hausdorff=0.0, false_negative_rate=0.0, false_positive_rate=0.0, false_discovery_rate=0.0, voi=0.0, mean_false_distance=0.0, mean_false_negative_distance=0.0, mean_false_positive_distance=0.0, mean_false_distance_clipped=0.0, mean_false_negative_distance_clipped=0.0, mean_false_positive_distance_clipped=0.0, precision_with_tolerance=0.0, recall_with_tolerance=0.0, f1_score_with_tolerance=0.0, precision=0.0, recall=0.0, f1_score=0.0)
        Note:
            This function is used to evaluate the output array against the evaluation array.
        """
        output_array = open_from_identifier(output_array_identifier)
        # removed the .squeeze() because it was used for batch size and now we are feeding 4d c, z, y, x
        evaluation_data = evaluation_array[evaluation_array.roi]
        output_data = output_array[output_array.roi]
        print(
            f"Evaluating binary segmentations on evaluation_data of shape: {evaluation_data.shape}"
        )
        assert (
            evaluation_data.shape == output_data.shape
        ), f"{evaluation_data.shape} vs {output_data.shape}"
        if "c^" in evaluation_array.axis_names and "c^" in output_array.axis_names:
            score_dict = []
            for indx, channel in enumerate(
                range(evaluation_array.shape[evaluation_array.axis_names.index("c^")])
            ):
                evaluation_channel_data = evaluation_data.take(
                    indices=indx, axis=evaluation_array.axis_names.index("c^")
                )
                output_channel_data = output_data.take(
                    indices=indx, axis=output_array.axis_names.index("c^")
                )
                evaluator = ArrayEvaluator(
                    evaluation_channel_data,
                    output_channel_data,
                    not evaluation_channel_data.any(),
                    not output_channel_data.any(),
                    metric_params={
                        "clip_distance": self.clip_distance,
                        "tol_distance": self.tol_distance,
                    },
                    resolution=evaluation_array.voxel_size,
                )
                score_dict.append(
                    (
                        f"{channel}",
                        BinarySegmentationEvaluationScores(
                            dice=evaluator.dice(),
                            jaccard=evaluator.jaccard(),
                            hausdorff=evaluator.hausdorff(),
                            false_negative_rate=evaluator.false_negative_rate(),
                            false_negative_rate_with_tolerance=evaluator.false_negative_rate_with_tolerance(),
                            false_positive_rate=evaluator.false_positive_rate(),
                            false_discovery_rate=evaluator.false_discovery_rate(),
                            false_positive_rate_with_tolerance=evaluator.false_positive_rate_with_tolerance(),
                            voi=evaluator.voi(),
                            mean_false_distance=evaluator.mean_false_distance(),
                            mean_false_negative_distance=evaluator.mean_false_negative_distance(),
                            mean_false_positive_distance=evaluator.mean_false_positive_distance(),
                            mean_false_distance_clipped=evaluator.mean_false_distance_clipped(),
                            mean_false_negative_distance_clipped=evaluator.mean_false_negative_distance_clipped(),
                            mean_false_positive_distance_clipped=evaluator.mean_false_positive_distance_clipped(),
                            precision_with_tolerance=evaluator.precision_with_tolerance(),
                            recall_with_tolerance=evaluator.recall_with_tolerance(),
                            f1_score_with_tolerance=evaluator.f1_score_with_tolerance(),
                            precision=evaluator.precision(),
                            recall=evaluator.recall(),
                            f1_score=evaluator.f1_score(),
                        ),
                    )
                )
            return MultiChannelBinarySegmentationEvaluationScores(score_dict)

        else:
            evaluator = ArrayEvaluator(
                evaluation_data,
                output_data,
                not evaluation_data.any(),
                not output_data.any(),
                metric_params={
                    "clip_distance": self.clip_distance,
                    "tol_distance": self.tol_distance,
                },
                resolution=evaluation_array.voxel_size,
            )
            return BinarySegmentationEvaluationScores(
                dice=evaluator.dice(),
                jaccard=evaluator.jaccard(),
                hausdorff=evaluator.hausdorff(),
                false_negative_rate=evaluator.false_negative_rate(),
                false_negative_rate_with_tolerance=evaluator.false_negative_rate_with_tolerance(),
                false_positive_rate=evaluator.false_positive_rate(),
                false_discovery_rate=evaluator.false_discovery_rate(),
                false_positive_rate_with_tolerance=evaluator.false_positive_rate_with_tolerance(),
                voi=evaluator.voi(),
                mean_false_distance=evaluator.mean_false_distance(),
                mean_false_negative_distance=evaluator.mean_false_negative_distance(),
                mean_false_positive_distance=evaluator.mean_false_positive_distance(),
                mean_false_distance_clipped=evaluator.mean_false_distance_clipped(),
                mean_false_negative_distance_clipped=evaluator.mean_false_negative_distance_clipped(),
                mean_false_positive_distance_clipped=evaluator.mean_false_positive_distance_clipped(),
                precision_with_tolerance=evaluator.precision_with_tolerance(),
                recall_with_tolerance=evaluator.recall_with_tolerance(),
                f1_score_with_tolerance=evaluator.f1_score_with_tolerance(),
                precision=evaluator.precision(),
                recall=evaluator.recall(),
                f1_score=evaluator.f1_score(),
            )

    @property
    def score(self):
        """
        Return the evaluation scores.

        Returns:
            BinarySegmentationEvaluationScores or MultiChannelBinarySegmentationEvaluationScores
                the evaluation scores
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> binary_segmentation_evaluator = BinarySegmentationEvaluator(clip_distance=200, tol_distance=40, channels=["channel1", "channel2"])
            >>> binary_segmentation_evaluator.score
            BinarySegmentationEvaluationScores(dice=0.0, jaccard=0.0, hausdorff=0.0, false_negative_rate=0.0, false_positive_rate=0.0, false_discovery_rate=0.0, voi=0.0, mean_false_distance=0.0, mean_false_negative_distance=0.0, mean_false_positive_distance=0.0, mean_false_distance_clipped=0.0, mean_false_negative_distance_clipped=0.0, mean_false_positive_distance_clipped=0.0, precision_with_tolerance=0.0, recall_with_tolerance=0.0, f1_score_with_tolerance=0.0, precision=0.0, recall=0.0, f1_score=0.0)
        Note:
            This function is used to return the evaluation scores.
        """
        channel_scores = []
        for channel in self.channels:
            channel_scores.append((channel, BinarySegmentationEvaluationScores()))
        return MultiChannelBinarySegmentationEvaluationScores(channel_scores)

    def _evaluate(self, output_data, evaluation_data, voxel_size):
        """
        Evaluate the output array against the evaluation array.

        Args:
            output_data : np.ndarray
                the output data
            evaluation_data : np.ndarray
                the evaluation data
            voxel_size : Tuple[float, float, float]
                the voxel size
        Returns:
            BinarySegmentationEvaluationScores
                the evaluation scores
        Examples:
            >>> binary_segmentation_evaluator = BinarySegmentationEvaluator(clip_distance=200, tol_distance=40, channels=["channel1", "channel2"])
            >>> output_data = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
            >>> evaluation_data = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
            >>> voxel_size = (1, 1, 1)
            >>> binary_segmentation_evaluator._evaluate(output_data, evaluation_data, voxel_size)
            BinarySegmentationEvaluationScores(dice=0.0, jaccard=0.0, hausdorff=0.0, false_negative_rate=0.0, false_positive_rate=0.0, false_discovery_rate=0.0, voi=0.0, mean_false_distance=0.0, mean_false_negative_distance=0.0, mean_false_positive_distance=0.0, mean_false_distance_clipped=0.0, mean_false_negative_distance_clipped=0.0, mean_false_positive_distance_clipped=0.0, precision_with_tolerance=0.0, recall_with_tolerance=0.0, f1_score_with_tolerance=0.0, precision=0.0, recall=0.0, f1_score=0.0)
        Note:
            This function is used to evaluate the output array against the evaluation array.
        """
        evaluator = ArrayEvaluator(
            evaluation_data,
            output_data,
            not evaluation_data.any(),
            not output_data.any(),
            metric_params={
                "clip_distance": self.clip_distance,
                "tol_distance": self.tol_distance,
            },
            resolution=voxel_size,
        )
        return BinarySegmentationEvaluationScores(
            dice=evaluator.dice(),
            jaccard=evaluator.jaccard(),
            hausdorff=evaluator.hausdorff(),
            false_negative_rate=evaluator.false_negative_rate(),
            false_negative_rate_with_tolerance=evaluator.false_negative_rate_with_tolerance(),
            false_positive_rate=evaluator.false_positive_rate(),
            false_discovery_rate=evaluator.false_discovery_rate(),
            false_positive_rate_with_tolerance=evaluator.false_positive_rate_with_tolerance(),
            voi=evaluator.voi(),
            mean_false_distance=evaluator.mean_false_distance(),
            mean_false_negative_distance=evaluator.mean_false_negative_distance(),
            mean_false_positive_distance=evaluator.mean_false_positive_distance(),
            mean_false_distance_clipped=evaluator.mean_false_distance_clipped(),
            mean_false_negative_distance_clipped=evaluator.mean_false_negative_distance_clipped(),
            mean_false_positive_distance_clipped=evaluator.mean_false_positive_distance_clipped(),
            precision_with_tolerance=evaluator.precision_with_tolerance(),
            recall_with_tolerance=evaluator.recall_with_tolerance(),
            f1_score_with_tolerance=evaluator.f1_score_with_tolerance(),
            precision=evaluator.precision(),
            recall=evaluator.recall(),
            f1_score=evaluator.f1_score(),
        )


class ArrayEvaluator:
    """
    Given a binary segmentation, compute various metrics to determine their similarity. The metrics include:
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
        truth : np.ndarray
            the truth binary segmentation
        test : np.ndarray
            the test binary segmentation
        truth_empty : bool
            whether the truth binary segmentation is empty
        test_empty : bool
            whether the test binary segmentation is empty
        cremieval : CremiEvaluator
            the cremi evaluator
        resolution : Tuple[float, float, float]
            the resolution
    Methods:
        dice
            Return the Dice coefficient.
        jaccard
            Return the Jaccard coefficient.
        hausdorff
            Return the Hausdorff distance.
        false_negative_rate
            Return the false negative rate.
        false_positive_rate
            Return the false positive rate.
        false_discovery_rate
            Return the false discovery rate.
        precision
            Return the precision.
        recall
            Return the recall.
        f1_score
            Return the F1 score.
        voi
            Return the VOI.
        mean_false_distance
            Return the mean false distance.
        mean_false_negative_distance
            Return the mean false negative distance.
        mean_false_positive_distance
            Return the mean false positive distance.
        mean_false_distance_clipped
            Return the mean false distance clipped.
        mean_false_negative_distance_clipped
            Return the mean false negative distance clipped.
        mean_false_positive_distance_clipped
            Return the mean false positive distance clipped.
        false_positive_rate_with_tolerance
            Return the false positive rate with tolerance.
        false_negative_rate_with_tolerance
            Return the false negative rate with tolerance.
        precision_with_tolerance
            Return the precision with tolerance.
        recall_with_tolerance
            Return the recall with tolerance.
        f1_score_with_tolerance
            Return the F1 score with tolerance.
    Note:
        The ArrayEvaluator class is used to evaluate the performance of a binary segmentation task.
        The class provides methods to evaluate the truth binary segmentation against the test binary segmentation.
        All evaluation scores should inherit from this class.
    """

    def __init__(
        self,
        truth_binary,
        test_binary,
        truth_empty,
        test_empty,
        metric_params,
        resolution,
    ):
        """
        Initialize the array evaluator.

        Args:
            truth_binary : np.ndarray
                the truth binary segmentation
            test_binary : np.ndarray
                the test binary segmentation
            truth_empty : bool
                whether the truth binary segmentation is empty
            test_empty : bool
                whether the test binary segmentation is empty
            metric_params : Dict[str, float]
                the metric parameters
            resolution : Tuple[float, float, float]
                the resolution
        Returns:
            ArrayEvaluator
                the array evaluator
        Raises:
            ValueError: if the truth binary segmentation is not valid
        Examples:
            >>> truth_binary = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
            >>> test_binary = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
            >>> truth_empty = False
            >>> test_empty = False
            >>> metric_params = {"clip_distance": 200, "tol_distance": 40}
            >>> resolution = (1, 1, 1)
            >>> array_evaluator = ArrayEvaluator(truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution)
        Note:
            This function is used to initialize the array evaluator.
        """
        self.truth = truth_binary.astype(np.uint8)
        self.test = test_binary.astype(np.uint8)
        self.truth_empty = truth_empty
        self.test_empty = test_empty
        self.cremieval = CremiEvaluator(
            truth_binary,
            test_binary,
            sampling=resolution,
            clip_distance=metric_params["clip_distance"],
            tol_distance=metric_params["tol_distance"],
        )
        self.resolution = resolution

    @lazy_property.LazyProperty
    def truth_itk(self):
        """
        A SimpleITK image of the truth binary segmentation.

        Returns:
            sitk.Image
                the truth binary segmentation as a SimpleITK image
        Examples:
            >>> array_evaluator = ArrayEvaluator(truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution)
            >>> array_evaluator.truth_itk
            <SimpleITK.SimpleITK.Image; proxy of <Swig Object of type 'std::vector< itk::simple::Image >::value_type *' at 0x7f8b1c0b3f30> >
        Note:
            This function is used to return the truth binary segmentation as a SimpleITK image.
        """
        res = sitk.GetImageFromArray(self.truth)
        res.SetSpacing(self.resolution)
        return res

    @lazy_property.LazyProperty
    def test_itk(self):
        """
        A SimpleITK image of the test binary segmentation.

        Args:
            test : np.ndarray
                the test binary segmentation
            resolution : Tuple[float, float, float]
                the resolution
        Returns:
            sitk.Image
                the test binary segmentation as a SimpleITK image
        Raises:
            ValueError: if the test binary segmentation is not valid
        Examples:
            >>> array_evaluator = ArrayEvaluator(truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution)
            >>> array_evaluator.test_itk
            <SimpleITK.SimpleITK.Image; proxy of <Swig Object of type 'std::vector< itk::simple::Image >::value_type *' at 0x7f8b1c0b3f30> >
        Note:
            This function is used to return the test binary segmentation as a SimpleITK image.
        """
        res = sitk.GetImageFromArray(self.test)
        res.SetSpacing(self.resolution)
        return res

    @lazy_property.LazyProperty
    def overlap_measures_filter(self):
        """
        A SimpleITK filter to compute overlap measures.

        Args:
            truth_itk : sitk.Image
                the truth binary segmentation as a SimpleITK image
            test_itk : sitk.Image
                the test binary segmentation as a SimpleITK image
        Returns:
            sitk.LabelOverlapMeasuresImageFilter
                the overlap measures filter
        Raises:
            ValueError: if the truth binary segmentation or the test binary segmentation is not valid
        Examples:
            >>> array_evaluator = ArrayEvaluator(truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution)
            >>> array_evaluator.overlap_measures_filter
            <SimpleITK.SimpleITK.LabelOverlapMeasuresImageFilter; proxy of <Swig Object of type 'itk::simple::LabelOverlapMeasuresImageFilter *' at 0x7f8b1c0b3f30> >
        Note:
            This function is used to return the overlap measures filter.
        """
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(self.test_itk, self.truth_itk)
        return overlap_measures_filter

    def dice(self):
        """
        The Dice coefficient.

        Args:
            truth_itk : sitk.Image
                the truth binary segmentation as a SimpleITK image
            test_itk : sitk.Image
                the test binary segmentation as a SimpleITK image
        Returns:
            float
                the Dice coefficient
        Raises:
            ValueError: if the truth binary segmentation or the test binary segmentation is not valid
        Examples:
            >>> array_evaluator = ArrayEvaluator(truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution)
            >>> array_evaluator.dice()
            0.0
        Note:
            This function is used to return the Dice coefficient.
        """
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetDiceCoefficient()
        else:
            return np.nan

    def jaccard(self):
        """
        The Jaccard coefficient.

        Args:
            truth_itk : sitk.Image
                the truth binary segmentation as a SimpleITK image
            test_itk : sitk.Image
                the test binary segmentation as a SimpleITK image
        Returns:
            float
                the Jaccard coefficient
        Raises:
            ValueError: if the truth binary segmentation or the test binary segmentation is not valid
        Examples:
            >>> array_evaluator = ArrayEvaluator(truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution)
            >>> array_evaluator.jaccard()
            0.0
        Note:
            This function is used to return the Jaccard coefficient.

        """
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetJaccardCoefficient()
        else:
            return np.nan

    def hausdorff(self):
        """
        The Hausdorff distance.

        Args:
            None
        Returns:
            float: the Hausdorff distance
        Raises:
            None
        Examples:
            >>> array_evaluator = ArrayEvaluator(truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution)
            >>> array_evaluator.hausdorff()
            0.0
        Note:
            This function is used to return the Hausdorff distance between the truth binary segmentation and the test binary segmentation.

            If either the truth or test binary segmentation is empty, the function returns 0.
            Otherwise, it calculates the Hausdorff distance using the HausdorffDistanceImageFilter from the SimpleITK library.
        """
        if self.truth_empty and self.test_empty:
            return 0
        elif not self.truth_empty and not self.test_empty:
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(self.test_itk, self.truth_itk)
            return hausdorff_distance_filter.GetHausdorffDistance()
        else:
            return np.nan

    def false_negative_rate(self):
        """
        The false negative rate.

        Returns:
            float
                the false negative rate
        Returns:
            ValueError: if the truth binary segmentation or the test binary segmentation is not valid
        Examples:
            >>> array_evaluator = ArrayEvaluator(truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution)
            >>> array_evaluator.false_negative_rate()
            0.0
        Note:
            This function is used to return the false negative rate.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.overlap_measures_filter.GetFalseNegativeError()

    def false_positive_rate(self):
        """
        The false positive rate.

        Args:
            truth_itk : sitk.Image
                the truth binary segmentation as a SimpleITK image
            test_itk : sitk.Image
                the test binary segmentation as a SimpleITK image
        Returns:
            float
                the false positive rate
        Raises:
            ValueError: if the truth binary segmentation or the test binary segmentation is not valid
        Examples:
            >>> array_evaluator = ArrayEvaluator(truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution)
            >>> array_evaluator.false_positive_rate()
            0.0
        Note:
            This function is used to return the false positive rate.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return (self.false_discovery_rate() * np.sum(self.test != 0)) / np.sum(
                self.truth == 0
            )

    def false_discovery_rate(self):
        """
        Calculate the false discovery rate (FDR) for the binary segmentation evaluation.

        Returns:
            float: The false discovery rate.
        Raises:
            None
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_discovery_rate()
            0.25
        Note:
            The false discovery rate is a measure of the proportion of false positives among the predicted positive samples.
            It is calculated as the ratio of false positives to the sum of true positives and false positives.
            If either the ground truth or the predicted segmentation is empty, the FDR is set to NaN.
        """
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetFalsePositiveError()
        else:
            return np.nan

    def precision(self):
        """
        Calculate the precision of the binary segmentation evaluation.

        Returns:
            float: The precision value.
        Raises:
            None.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.precision()
            0.75
        Note:
            Precision is a measure of the accuracy of the positive predictions made by the model.
            It is calculated as the ratio of true positives to the total number of positive predictions.
            If either the ground truth or the predicted values are empty, the precision value will be NaN.
        """

        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            pred_pos = np.sum(self.test != 0)
            tp = pred_pos - (self.false_discovery_rate() * pred_pos)
            return float(np.float32(tp) / np.float32(pred_pos))

    def recall(self):
        """
        Calculate the recall metric for binary segmentation evaluation.

        Returns:
            float: The recall value.
        Raises:
            None
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.recall()
            0.75
        Note:
            Recall is a measure of the ability of a binary classifier to identify all positive samples.
            It is calculated as the ratio of true positives to the total number of actual positives.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            cond_pos = np.sum(self.truth != 0)
            tp = cond_pos - (self.false_negative_rate() * cond_pos)
            return float(np.float32(tp) / np.float32(cond_pos))

    def f1_score(self):
        """
        Calculate the F1 score for binary segmentation evaluation.

        Returns:
            float: The F1 score value.
        Raises:
            None.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.f1_score()
            0.75
        Note:
            The F1 score is the harmonic mean of precision and recall.
            It is a measure of the balance between precision and recall, providing a single metric to evaluate the model's performance.

            If either the ground truth or the predicted values are empty, the F1 score will be NaN.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            prec = self.precision()
            rec = self.recall()
            if prec == 0 and rec == 0:
                return np.nan
            else:
                return 2 * (rec * prec) / (rec + prec)

    def voi(self):
        """
        Calculate the Variation of Information (VOI) for binary segmentation evaluation.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            float: The VOI value.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.voi()
            0.75
        Note:
            The VOI is a measure of the similarity between two segmentations.
            It combines the split and merge errors into a single measure of segmentation quality.
            If either the ground truth or the predicted values are empty, the VOI will be NaN.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            voi_split, voi_merge = voi(
                self.test + 1, self.truth + 1, ignore_groundtruth=[]
            )
            return voi_split + voi_merge

    def mean_false_distance(self):
        """
        Calculate the mean false distance between the ground truth and the test results.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            float: The mean false distance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.mean_false_distance()
            0.25
        Note:
            - This method returns np.nan if either the ground truth or the test results are empty.
            - The mean false distance is a measure of the average distance between the false positive pixels in the test results and the nearest true positive pixels in the ground truth.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_distance

    def mean_false_negative_distance(self):
        """
        Calculate the mean false negative distance between the ground truth and the test results.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            float: The mean false negative distance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.mean_false_negative_distance()
            0.25
        Note:
            This method returns np.nan if either the ground truth or the test results are empty.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_negative_distance

    def mean_false_positive_distance(self):
        """
        Calculate the mean false positive distance.

        This method calculates the mean false positive distance between the ground truth and the test results.
        If either the ground truth or the test results are empty, the method returns NaN.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            float: The mean false positive distance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.mean_false_positive_distance()
            0.5
        Note:
            The mean false positive distance is a measure of the average distance between false positive pixels in the
            test results and the corresponding ground truth pixels. It is commonly used to evaluate the performance of
            binary segmentation algorithms.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_positive_distance

    def mean_false_distance_clipped(self):
        """
        Calculate the mean false distance (clipped) between the ground truth and the test results.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            float: The mean false distance (clipped) value.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.mean_false_distance_clipped()
            0.123
        Note:
            This method returns np.nan if either the ground truth or the test results are empty.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_distance_clipped

    def mean_false_negative_distance_clipped(self):
        """
        Calculate the mean false negative distance, with clipping.

        This method calculates the mean false negative distance between the ground truth and the test results.
        The distance is clipped to avoid extreme values.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            float: The mean false negative distance with clipping.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.mean_false_negative_distance_clipped()
            0.123
        Note:
            - The mean false negative distance is a measure of the average distance between the false negative pixels in the ground truth and the test results.
            - Clipping the distance helps to avoid extreme values that may skew the overall evaluation.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_negative_distances_clipped

    def mean_false_positive_distance_clipped(self):
        """
        Calculate the mean false positive distance, with clipping.

        This method calculates the mean false positive distance between the ground truth and the test results,
        taking into account any clipping that may have been applied.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            float: The mean false positive distance with clipping.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.mean_false_positive_distance_clipped()
            0.25
        Note:
            - The mean false positive distance is a measure of the average distance between false positive pixels
              in the test results and the corresponding ground truth pixels.
            - If either the ground truth or the test results are empty, the method returns NaN.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_positive_distances_clipped

    def false_positive_rate_with_tolerance(self):
        """
        Calculate the false positive rate with tolerance.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            float: The false positive rate with tolerance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_positive_rate_with_tolerance()
            0.25
        Note:
            This method calculates the false positive rate with tolerance by comparing the truth and test data.
            If either the truth or test data is empty, it returns NaN.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.false_positive_rate_with_tolerance

    def false_negative_rate_with_tolerance(self):
        """
        Calculate the false negative rate with tolerance.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            The false negative rate with tolerance as a floating-point number.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_negative_rate_with_tolerance()
            0.25
        Note:
            This method calculates the false negative rate with tolerance, which is a measure of the proportion of false negatives in a binary segmentation evaluation. If either the ground truth or the test data is empty, the method returns NaN.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.false_negative_rate_with_tolerance

    def precision_with_tolerance(self):
        """
        Calculate the precision with tolerance.

        This method calculates the precision with tolerance by comparing the truth and test data.
        Precision is the ratio of true positives to the sum of true positives and false positives.
        Tolerance is a distance threshold within which two pixels are considered to be a match.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            float: The precision with tolerance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.precision_with_tolerance()
            0.75
        Note:
            - Precision is a measure of the accuracy of the positive predictions.
            - If either the ground truth or the test data is empty, the method returns NaN.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.precision_with_tolerance

    def recall_with_tolerance(self):
        """
        Calculate the recall with tolerance for the binary segmentation evaluator.

        Returns:
            float: The recall with tolerance value.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.recall_with_tolerance()
            0.75
        Note:
            This method calculates the recall with tolerance, which is a measure of how well the binary segmentation evaluator performs. It returns the recall with tolerance value as a float. If either the truth or test data is empty, it returns NaN.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.recall_with_tolerance

    def f1_score_with_tolerance(self):
        """
        Calculate the F1 score with tolerance.

        This method calculates the F1 score with tolerance between the ground truth and the test results.
        If either the ground truth or the test results are empty, the function returns NaN.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
        Returns:
            float: The F1 score with tolerance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.f1_score_with_tolerance()
            0.85
        Note:
            The F1 score is a measure of a test's accuracy. It considers both the precision and recall of the test to compute the score.
            The tolerance parameter allows for a certain degree of variation between the ground truth and the test results.
        """
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.f1_score_with_tolerance


class CremiEvaluator:
    """
    Evaluate the performance of a binary segmentation task using the CREMI score.
    The CREMI score is a measure of the similarity between two binary segmentations.

    Attributes:
        truth : np.ndarray
            the truth binary segmentation
        test : np.ndarray
            the test binary segmentation
        sampling : Tuple[float, float, float]
            the sampling resolution
        clip_distance : float
            the maximum distance to clip
        tol_distance : float
            the tolerance distance
    Methods:
        false_positive_distances
            Return the false positive distances.
        false_positives_with_tolerance
            Return the false positives with tolerance.
        false_positive_rate_with_tolerance
            Return the false positive rate with tolerance.
        false_negatives_with_tolerance
            Return the false negatives with tolerance.
        false_negative_rate_with_tolerance
            Return the false negative rate with tolerance.
        true_positives_with_tolerance
            Return the true positives with tolerance.
        precision_with_tolerance
            Return the precision with tolerance.
        recall_with_tolerance
            Return the recall with tolerance.
        f1_score_with_tolerance
            Return the F1 score with tolerance.
        mean_false_positive_distances_clipped
            Return the mean false positive distances clipped.
        mean_false_negative_distances_clipped
            Return the mean false negative distances clipped.
        mean_false_positive_distance
            Return the mean false positive distance.
        false_negative_distances
            Return the false negative distances.
        mean_false_negative_distance
            Return the mean false negative distance.
        mean_false_distance
            Return the mean false distance.
        mean_false_distance_clipped
            Return the mean false distance clipped.
    Note:
        - The CremiEvaluator class is used to evaluate the performance of a binary segmentation task using the CREMI score.
        - True and test binary segmentations are compared to calculate various evaluation metrics.
        - The class provides methods to evaluate the performance of the binary segmentation task.
        - Toleration distance is used to determine the tolerance level for the evaluation.
        - Clip distance is used to clip the distance values to avoid extreme values.
        - All evaluation scores should inherit from this class.
    """

    def __init__(
        self, truth, test, sampling=(1, 1, 1), clip_distance=200, tol_distance=40
    ):
        """
        Initialize the Cremi evaluator.

        Args:
            truth : np.ndarray
                the truth binary segmentation
            test : np.ndarray
                the test binary segmentation
            sampling : Tuple[float, float, float]
                the sampling resolution
            clip_distance : float
                the maximum distance to clip
            tol_distance : float
                the tolerance distance
        Returns:
            CremiEvaluator
                the Cremi evaluator
        Raises:
            ValueError: if the truth binary segmentation is not valid
        Examples:
            >>> truth = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
            >>> test = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
            >>> sampling = (1, 1, 1)
            >>> clip_distance = 200
            >>> tol_distance = 40
            >>> cremi_evaluator = CremiEvaluator(truth, test, sampling, clip_distance, tol_distance)
        Note:
            This function is used to initialize the Cremi evaluator.
        """
        self.test = test
        self.truth = truth
        self.sampling = sampling
        self.clip_distance = clip_distance
        self.tol_distance = tol_distance

    @lazy_property.LazyProperty
    def test_mask(self):
        """
        Generate a binary mask for the test data.

        Args:
            test : np.ndarray
                the test binary segmentation
        Returns:
            test_mask (ndarray): A binary mask indicating the regions of interest in the test data.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.test = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            >>> evaluator.test_mask()
            array([[False,  True, False],
                    [ True,  True,  True],
                    [False,  True, False]])
        Note:
            This method assumes that the background class is represented by the constant `BG`.
        """
        # todo: more involved masking
        test_mask = self.test == BG
        return test_mask

    @lazy_property.LazyProperty
    def truth_mask(self):
        """
        Returns a binary mask indicating the truth values.

        Args:
            truth : np.ndarray
                the truth binary segmentation
        Returns:
            truth_mask (ndarray): A binary mask where True indicates the truth values and False indicates other values.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> mask = evaluator.truth_mask()
            >>> print(mask)
            [[ True  True False]
                [False  True False]
                [ True False False]]
        Note:
            The truth mask is computed by comparing the truth values with a predefined background value (BG).
        """
        truth_mask = self.truth == BG
        return truth_mask

    @lazy_property.LazyProperty
    def test_edt(self):
        """
        Calculate the Euclidean Distance Transform (EDT) of the test mask.

        Args:
            self.test_mask (ndarray): The binary test mask.
            self.sampling (float or sequence of floats): The pixel spacing or sampling along each dimension.
        Returns:
            ndarray: The Euclidean Distance Transform of the test mask.
        Examples:
            # Example 1:
            test_mask = np.array([[0, 0, 1],
                                  [1, 1, 1],
                                  [0, 0, 0]])
            sampling = 1.0
            result = test_edt(test_mask, sampling)
            # Output: array([[1.        , 1.        , 0.        ],
            #                [0.        , 0.        , 0.        ],
            #                [1.        , 1.        , 1.41421356]])

            # Example 2:
            test_mask = np.array([[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]])
            sampling = 0.5
            result = test_edt(test_mask, sampling)
            # Output: array([[0.5       , 0.        , 0.5       ],
            #                [0.        , 0.70710678, 0.        ],
            #                [0.5       , 0.        , 0.5       ]])

        Note:
            The Euclidean Distance Transform (EDT) calculates the distance from each pixel in the binary mask to the nearest boundary pixel. It is commonly used in image processing and computer vision tasks, such as edge detection and shape analysis.
        """
        test_edt = scipy.ndimage.distance_transform_edt(self.test_mask, self.sampling)
        return test_edt

    @lazy_property.LazyProperty
    def truth_edt(self):
        """
        Calculate the Euclidean Distance Transform (EDT) of the ground truth mask.

        Args:
            self.truth_mask (ndarray): The binary ground truth mask.
            self.sampling (float or sequence of floats): The pixel spacing or sampling along each dimension.
        Returns:
            ndarray: The Euclidean Distance Transform of the ground truth mask.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> edt = evaluator.truth_edt()
        Note:
            The Euclidean Distance Transform (EDT) calculates the distance from each pixel in the binary mask to the nearest boundary pixel. It is commonly used in image processing and computer vision tasks.
        """
        truth_edt = scipy.ndimage.distance_transform_edt(self.truth_mask, self.sampling)
        return truth_edt

    @lazy_property.LazyProperty
    def false_positive_distances(self):
        """
        Calculate the distances of false positive pixels from the ground truth segmentation.

        Args:
            self.test_mask (ndarray): The binary test mask.
            self.truth_edt (ndarray): The Euclidean Distance Transform of the ground truth segmentation.
        Returns:
            numpy.ndarray: An array containing the distances of false positive pixels from the ground truth segmentation.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> distances = evaluator.false_positive_distances()
            >>> print(distances)
            [1.2, 0.8, 2.5, 1.0]
        Note:
            This method assumes that the ground truth segmentation and the test mask have been initialized.
            The ground truth segmentation is stored in the `truth_edt` attribute, and the test mask is obtained by inverting the `test_mask` attribute.
        """
        test_bin = np.invert(self.test_mask)
        false_positive_distances = self.truth_edt[test_bin]
        return false_positive_distances

    @lazy_property.LazyProperty
    def false_positives_with_tolerance(self):
        """
        Calculate the number of false positives with a given tolerance distance.

        Args:
            self.false_positive_distances (ndarray): The distances of false positive pixels from the ground truth segmentation.
            self.tol_distance (float): The tolerance distance.
        Returns:
            int: The number of false positives with a distance greater than the tolerance distance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_positive_distances = [1, 2, 3]
            >>> evaluator.tol_distance = 2
            >>> false_positives = evaluator.false_positives_with_tolerance()
            >>> print(false_positives)
            1
        Note:
            The `false_positive_distances` attribute should be initialized before calling this method.

        """
        return np.sum(self.false_positive_distances > self.tol_distance)

    @lazy_property.LazyProperty
    def false_positive_rate_with_tolerance(self):
        """
        Calculate the false positive rate with tolerance.

        This method calculates the false positive rate by dividing the number of false positives with tolerance
        by the number of condition negatives.

        Args:
            self.false_positives_with_tolerance (int): The number of false positives with tolerance.
            self.truth_mask (ndarray): The binary ground truth mask.
        Returns:
            float: The false positive rate with tolerance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_positives_with_tolerance = 10
            >>> evaluator.truth_mask = np.array([0, 1, 0, 1, 0])
            >>> evaluator.false_positive_rate_with_tolerance()
            0.5
        Note:
            The false positive rate with tolerance is a measure of the proportion of false positive predictions
            with respect to the total number of condition negatives. It is commonly used in binary segmentation tasks.
        """
        condition_negative = np.sum(self.truth_mask)
        return float(
            np.float32(self.false_positives_with_tolerance)
            / np.float32(condition_negative)
        )

    @lazy_property.LazyProperty
    def false_negatives_with_tolerance(self):
        """
        Calculate the number of false negatives with tolerance.

        Args:
            self.false_negative_distances (ndarray): The distances of false negative pixels from the ground truth segmentation.
            self.tol_distance (float): The tolerance distance.
        Returns:
            int: The number of false negatives with tolerance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_negative_distances = [1, 2, 3]
            >>> evaluator.tol_distance = 2
            >>> false_negatives = evaluator.false_negatives_with_tolerance()
            >>> print(false_negatives)
            1
        Note:
            False negatives are cases where the model incorrectly predicts the absence of a positive class.
            The tolerance distance is used to determine whether a false negative is within an acceptable range.

        """
        return np.sum(self.false_negative_distances > self.tol_distance)

    @lazy_property.LazyProperty
    def false_negative_rate_with_tolerance(self):
        """
        Calculate the false negative rate with tolerance.

        This method calculates the false negative rate by dividing the number of false negatives
        with tolerance by the number of condition positives.

        Args:
            self.false_negatives_with_tolerance (int): The number of false negatives with tolerance.
            self.false_negative_distances (ndarray): The distances of false negative pixels from the ground truth segmentation.
        Returns:
            float: The false negative rate with tolerance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_negative_distances = [1, 2, 3]
            >>> evaluator.false_negatives_with_tolerance = 2
            >>> evaluator.false_negative_rate_with_tolerance()
            0.6666666666666666
        Note:
            The false negative rate with tolerance is a measure of the proportion of condition positives
            that are incorrectly classified as negatives, considering a certain tolerance level.
        """
        condition_positive = len(self.false_negative_distances)
        return float(
            np.float32(self.false_negatives_with_tolerance)
            / np.float32(condition_positive)
        )

    @lazy_property.LazyProperty
    def true_positives_with_tolerance(self):
        """
        Calculate the number of true positives with tolerance.

        Args:
            self.test_mask (ndarray): The test binary segmentation mask.
            self.truth_mask (ndarray): The ground truth binary segmentation mask.
            self.false_negatives_with_tolerance (int): The number of false negatives with tolerance.
            self.false_positives_with_tolerance (int): The number of false positives with tolerance.
        Returns:
            int: The number of true positives with tolerance.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.test_mask = np.array([[0, 1], [1, 0]])
            >>> evaluator.truth_mask = np.array([[0, 1], [1, 0]])
            >>> evaluator.false_negatives_with_tolerance = 1
            >>> evaluator.false_positives_with_tolerance = 1
            >>> true_positives = evaluator.true_positives_with_tolerance()
            >>> print(true_positives)
            2
        Note:
            True positives are cases where the model correctly predicts the presence of a positive class.
            The tolerance distance is used to determine whether a true positive is within an acceptable range.
        """
        all_pos = np.sum(np.invert(self.test_mask & self.truth_mask))
        return (
            all_pos
            - self.false_negatives_with_tolerance
            - self.false_positives_with_tolerance
        )

    @lazy_property.LazyProperty
    def precision_with_tolerance(self):
        """
        Calculate the precision with tolerance.

        This method calculates the precision with tolerance by dividing the number of true positives
        with tolerance by the sum of true positives with tolerance and false positives with tolerance.

        Args:
            self.true_positives_with_tolerance (int): The number of true positives with tolerance.
            self.false_positives_with_tolerance (int): The number of false positives with tolerance.
        Returns:
            float: The precision with tolerance.
        Raises:
            ZeroDivisionError: If the sum of true positives with tolerance and false positives with tolerance is zero.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.true_positives_with_tolerance = 10
            >>> evaluator.false_positives_with_tolerance = 5
            >>> evaluator.precision_with_tolerance()
            0.6666666666666666
        Note:
            The precision with tolerance is a measure of the proportion of true positives with tolerance
            out of the total number of predicted positives with tolerance.
            It indicates how well the binary segmentation evaluator performs in terms of correctly identifying positive samples.
            If the sum of true positives with tolerance and false positives with tolerance is zero, the precision with tolerance is undefined and a ZeroDivisionError is raised.
        """
        return float(
            np.float32(self.true_positives_with_tolerance)
            / np.float32(
                self.true_positives_with_tolerance + self.false_positives_with_tolerance
            )
        )

    @lazy_property.LazyProperty
    def recall_with_tolerance(self):
        """
        A measure of the ability of a binary classifier to identify all positive samples.

        Args:
            self.true_positives_with_tolerance (int): The number of true positives with tolerance.
            self.false_negatives_with_tolerance (int): The number of false negatives with tolerance.
        Returns:
            float: The recall with tolerance value.
        Raises:
            ZeroDivisionError: If the sum of true positives with tolerance and false negatives with tolerance is zero.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.recall_with_tolerance()
            0.75
        Note:
            This method calculates the recall with tolerance, which is a measure of how well the binary segmentation evaluator performs. It returns the recall with tolerance value as a float. If either the truth or test data is empty, it returns NaN.
        """
        return float(
            np.float32(self.true_positives_with_tolerance)
            / np.float32(
                self.true_positives_with_tolerance + self.false_negatives_with_tolerance
            )
        )

    @lazy_property.LazyProperty
    def f1_score_with_tolerance(self):
        """
        Calculate the F1 score with tolerance.

        Args:
            self.recall_with_tolerance (float): The recall with tolerance value.
            self.precision_with_tolerance (float): The precision with tolerance value.
        Returns:
            float: The F1 score with tolerance.
        Raises:
            ZeroDivisionError: If both the recall with tolerance and precision with tolerance are zero.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.recall_with_tolerance = 0.8
            >>> evaluator.precision_with_tolerance = 0.9
            >>> evaluator.f1_score_with_tolerance()
            0.8571428571428571
        Note:
            The F1 score is a measure of a test's accuracy. It considers both the precision and recall of the test to compute the score.
            The F1 score with tolerance is calculated using the formula:
            F1 = 2 * (recall_with_tolerance * precision_with_tolerance) / (recall_with_tolerance + precision_with_tolerance)
            If both recall_with_tolerance and precision_with_tolerance are 0, the F1 score with tolerance will be NaN.
        """
        if self.recall_with_tolerance == 0 and self.precision_with_tolerance == 0:
            return np.nan
        else:
            return (
                2
                * (self.recall_with_tolerance * self.precision_with_tolerance)
                / (self.recall_with_tolerance + self.precision_with_tolerance)
            )

    @lazy_property.LazyProperty
    def mean_false_positive_distances_clipped(self):
        """
        Calculate the mean of the false positive distances, clipped to a maximum distance.

        Args:
            self.false_positive_distances (ndarray): The distances of false positive pixels from the ground truth segmentation.
            self.clip_distance (float): The maximum distance to clip.
        Returns:
            float: The mean of the false positive distances, clipped to a maximum distance.
        Raises:
            ValueError: If the clip distance is not set.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_positive_distances = [1, 2, 3, 4, 5]
            >>> evaluator.clip_distance = 3
            >>> evaluator.mean_false_positive_distances_clipped()
            2.5
        Note:

            This method calculates the mean of the false positive distances, where the distances are clipped to a maximum distance. The `false_positive_distances` attribute should be set before calling this method. The `clip_distance` attribute determines the maximum distance to which the distances are clipped.
        """
        mean_false_positive_distance_clipped = np.mean(
            np.clip(self.false_positive_distances, None, self.clip_distance)
        )
        return mean_false_positive_distance_clipped

    @lazy_property.LazyProperty
    def mean_false_negative_distances_clipped(self):
        """
        Calculate the mean of the false negative distances, clipped to a maximum distance.

        Args:
            self.false_negative_distances (ndarray): The distances of false negative pixels from the ground truth segmentation.
            self.clip_distance (float): The maximum distance to clip.
        Returns:
            float: The mean of the false negative distances, clipped to a maximum distance.
        Raises:
            ValueError: If the clip distance is not set.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_negative_distances = [1, 2, 3, 4, 5]
            >>> evaluator.clip_distance = 3
            >>> evaluator.mean_false_negative_distances_clipped()
            2.5
        Note:
            This method calculates the mean of the false negative distances, where the distances are clipped to a maximum distance. The `false_negative_distances` attribute should be set before calling this method. The `clip_distance` attribute determines the maximum distance to which the distances are clipped.
        """
        mean_false_negative_distance_clipped = np.mean(
            np.clip(self.false_negative_distances, None, self.clip_distance)
        )
        return mean_false_negative_distance_clipped

    @lazy_property.LazyProperty
    def mean_false_positive_distance(self):
        """
        Calculate the mean false positive distance.

        This method calculates the mean distance between the false positive points and the ground truth points.

        Args:
            self.false_positive_distances (ndarray): The distances of false positive pixels from the ground truth mask.
        Returns:
            float: The mean false positive distance.
        Raises:
            ValueError: If the false positive distances are not set.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_positive_distances = [1.2, 3.4, 2.1]
            >>> evaluator.mean_false_positive_distance()
            2.2333333333333334
        Note:
            The false positive distances should be set before calling this method using the `false_positive_distances` attribute.
        """
        mean_false_positive_distance = np.mean(self.false_positive_distances)
        return mean_false_positive_distance

    @lazy_property.LazyProperty
    def false_negative_distances(self):
        """
        Calculate the distances of false negative pixels from the ground truth mask.

        Args:
            self.truth_mask (ndarray): The binary ground truth mask.
        Returns:
            numpy.ndarray: An array containing the distances of false negative pixels.
        Raises:
            ValueError: If the truth mask is not set.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> distances = evaluator.false_negative_distances()
            >>> print(distances)
            [0.5, 1.0, 1.5, 2.0]
        Note:
            This method assumes that the ground truth mask and the test mask have already been set.
        """
        truth_bin = np.invert(self.truth_mask)
        false_negative_distances = self.test_edt[truth_bin]
        return false_negative_distances

    @lazy_property.LazyProperty
    def mean_false_negative_distance(self):
        """
        Calculate the mean false negative distance.

        Args:
            self.false_negative_distances (ndarray): The distances of false negative pixels from the ground truth mask.
        Returns:
            float: The mean false negative distance.
        Raises:
            ValueError: If the false negative distances are not set.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.false_negative_distances = [1.2, 3.4, 2.1]
            >>> evaluator.mean_false_negative_distance()
            2.2333333333333334
        Note:
            The mean false negative distance is calculated as the average of all false negative distances.

        """
        mean_false_negative_distance = np.mean(self.false_negative_distances)
        return mean_false_negative_distance

    @lazy_property.LazyProperty
    def mean_false_distance(self):
        """
        Calculate the mean false distance.

        This method calculates the mean false distance by taking the average of the mean false positive distance
        and the mean false negative distance.

        Args:
            self.mean_false_positive_distance (float): The mean false positive distance.
            self.mean_false_negative_distance (float): The mean false negative distance.
        Returns:
            float: The calculated mean false distance.
        Raises:
            ValueError: If the mean false positive distance or the mean false negative distance is not set.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.mean_false_distance()
            5.0
        Note:
            The mean false distance is a metric used to evaluate the performance of a binary segmentation model.
            It provides a measure of the average distance between false positive and false negative predictions.

        """
        mean_false_distance = 0.5 * (
            self.mean_false_positive_distance + self.mean_false_negative_distance
        )
        return mean_false_distance

    @lazy_property.LazyProperty
    def mean_false_distance_clipped(self):
        """
        Calculates the mean false distance clipped.

        This method calculates the mean false distance clipped by taking the average of the mean false positive distances
        clipped and the mean false negative distances clipped.

        Args:
            self.mean_false_positive_distances_clipped (float): The mean false positive distances clipped.
            self.mean_false_negative_distances_clipped (float): The mean false negative distances clipped.
        Returns:
            float: The calculated mean false distance clipped.
        Raises:
            ValueError: If the mean false positive distances clipped or the mean false negative distances clipped are not set.
        Examples:
            >>> evaluator = BinarySegmentationEvaluator()
            >>> evaluator.mean_false_distance_clipped()
            2.5
        Note:
            The mean false distance clipped is calculated as 0.5 * (mean_false_positive_distances_clipped +
            mean_false_negative_distances_clipped).

        """
        mean_false_distance_clipped = 0.5 * (
            self.mean_false_positive_distances_clipped
            + self.mean_false_negative_distances_clipped
        )
        return mean_false_distance_clipped

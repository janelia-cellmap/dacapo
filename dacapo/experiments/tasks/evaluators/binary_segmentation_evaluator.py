from .evaluator import Evaluator
from .binary_segmentation_evaluation_scores import (
    BinarySegmentationEvaluationScores,
    MultiChannelBinarySegmentationEvaluationScores,
)

from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
from dacapo.experiments.datasplits.datasets import Dataset
from dacapo.experiments.tasks.post_processors import PostProcessorParameters

import xarray as xr
import numpy as np
import SimpleITK as sitk
import cremi.evaluation
import lazy_property
import scipy

import itertools
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

BG = 0


class BinarySegmentationEvaluator(Evaluator):
    """
    Given a binary segmentation, compute various metrics to determine their similarity.
    """

    criteria = ["jaccard", "voi"]

    def __init__(self, clip_distance, tol_distance, channels):
        self.clip_distance = clip_distance
        self.tol_distance = tol_distance
        self.channels = channels
        self.criteria = [
            f"{channel}__{criteria}"
            for channel, criteria in itertools.product(channels, self.criteria)
        ]

    def evaluate(self, output_array_identifier, evaluation_array):
        output_array = ZarrArray.open_from_array_identifier(output_array_identifier)
        evaluation_data = evaluation_array[evaluation_array.roi]
        output_data = output_array[output_array.roi]
        logger.info(
            f"Evaluating binary segmentations on evaluation_data of shape: {evaluation_data.shape}"
        )
        assert (
            evaluation_data.shape == output_data.shape
        ), f"{evaluation_data.shape} vs {output_data.shape}"
        if "c" in evaluation_array.axes:
            score_dict = []
            for indx, channel in enumerate(evaluation_array.channels):
                evaluation_channel_data = evaluation_data.take(
                    indices=indx, axis=evaluation_array.axes.index("c")
                )
                output_channel_data = output_data.take(
                    indices=indx, axis=output_array.axes.index("c")
                )
                evaluator = Evaluator(
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
            evaluator = Evaluator(
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
        channel_scores = []
        for channel in self.channels:
            channel_scores.append((channel, BinarySegmentationEvaluationScores()))
        return MultiChannelBinarySegmentationEvaluationScores(channel_scores)

    def _evaluate(self, output_data, evaluation_data, voxel_size):
        evaluator = Evaluator(
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

    def is_best(
        self,
        dataset: Dataset,
        parameter: PostProcessorParameters,
        criterion: str,
        score: float,
    ):
        """
        Check if the provided score is the best for this dataset/parameter/criterion combo
        """
        if self._best_scores[(dataset, parameter, criterion)] is None:
            return True
        else:
            return score.is_better(self._best_scores, criterion)

    def set_best(self, validation_scores):
        """
        Find the best iteration for each dataset/post_processing_parameter/criterion
        """
        self._best_scores = {}
        scores = validation_scores.to_xarray()
        for dataset, parameters, criterion in itertools.product(
            scores.coords["datasets"].values,
            scores.coords["parameters"].values,
            scores.coords["criteria"].values,
        ):
            iteration_scores = scores.sel(
                datasets=[dataset], parameters=[parameters], criteria=[criterion]
            )
            if iteration_scores.size == 0:
                self._best_scores[(dataset, parameters, criterion)] = None
            else:
                # compute best
                winner = validation_scores.best(iteration_scores)[0]
                self._best_scores[
                    (
                        winner.coords["datasets"].item(),
                        winner.coords["parameters"].item(),
                        winner.coords["criteria"].item(),
                    )
                ] = (winner.coords["iteration"].item(), winner.item())



class Evaluator:
    def __init__(
        self,
        truth_binary,
        test_binary,
        truth_empty,
        test_empty,
        metric_params,
        resolution,
    ):
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
        res = sitk.GetImageFromArray(self.truth)
        res.SetSpacing(self.resolution)
        return res

    @lazy_property.LazyProperty
    def test_itk(self):
        res = sitk.GetImageFromArray(self.test)
        res.SetSpacing(self.resolution)
        return res

    @lazy_property.LazyProperty
    def overlap_measures_filter(self):
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(self.test_itk, self.truth_itk)
        return overlap_measures_filter

    def dice(self):
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetDiceCoefficient()
        else:
            return np.nan

    def jaccard(self):
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetJaccardCoefficient()
        else:
            return np.nan

    def hausdorff(self):
        if self.truth_empty and self.test_empty:
            return 0
        elif not self.truth_empty and not self.test_empty:
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(self.test_itk, self.truth_itk)
            return hausdorff_distance_filter.GetHausdorffDistance()
        else:
            return np.nan

    def false_negative_rate(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.overlap_measures_filter.GetFalseNegativeError()

    def false_positive_rate(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return (self.false_discovery_rate() * np.sum(self.test != 0)) / np.sum(
                self.truth == 0
            )

    def false_discovery_rate(self):
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetFalsePositiveError()
        else:
            return np.nan

    def precision(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            pred_pos = np.sum(self.test != 0)
            tp = pred_pos - (self.false_discovery_rate() * pred_pos)
            return float(tp) / float(pred_pos)

    def recall(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            cond_pos = np.sum(self.truth != 0)
            tp = cond_pos - (self.false_negative_rate() * cond_pos)
            return float(tp) / float(cond_pos)

    def f1_score(self):
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
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            voi_split, voi_merge = cremi.evaluation.voi(
                self.test + 1, self.truth + 1, ignore_groundtruth=[]
            )
            return voi_split + voi_merge

    def mean_false_distance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_distance

    def mean_false_negative_distance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_negative_distance

    def mean_false_positive_distance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_positive_distance

    def mean_false_distance_clipped(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_distance_clipped

    def mean_false_negative_distance_clipped(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_negative_distances_clipped

    def mean_false_positive_distance_clipped(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_positive_distances_clipped

    def false_positive_rate_with_tolerance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.false_positive_rate_with_tolerance

    def false_negative_rate_with_tolerance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.false_negative_rate_with_tolerance

    def precision_with_tolerance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.precision_with_tolerance

    def recall_with_tolerance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.recall_with_tolerance

    def f1_score_with_tolerance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.f1_score_with_tolerance


class CremiEvaluator:
    def __init__(
        self, truth, test, sampling=(1, 1, 1), clip_distance=200, tol_distance=40
    ):
        self.test = test
        self.truth = truth
        self.sampling = sampling
        self.clip_distance = clip_distance
        self.tol_distance = tol_distance

    @lazy_property.LazyProperty
    def test_mask(self):
        # todo: more involved masking
        test_mask = self.test == BG
        return test_mask

    @lazy_property.LazyProperty
    def truth_mask(self):
        truth_mask = self.truth == BG
        return truth_mask

    @lazy_property.LazyProperty
    def test_edt(self):
        test_edt = scipy.ndimage.distance_transform_edt(self.test_mask, self.sampling)
        return test_edt

    @lazy_property.LazyProperty
    def truth_edt(self):
        truth_edt = scipy.ndimage.distance_transform_edt(self.truth_mask, self.sampling)
        return truth_edt

    @lazy_property.LazyProperty
    def false_positive_distances(self):
        test_bin = np.invert(self.test_mask)
        false_positive_distances = self.truth_edt[test_bin]
        return false_positive_distances

    @lazy_property.LazyProperty
    def false_positives_with_tolerance(self):
        return np.sum(self.false_positive_distances > self.tol_distance)

    @lazy_property.LazyProperty
    def false_positive_rate_with_tolerance(self):
        condition_negative = np.sum(self.truth_mask)
        return float(self.false_positives_with_tolerance) / float(condition_negative)

    @lazy_property.LazyProperty
    def false_negatives_with_tolerance(self):
        return np.sum(self.false_negative_distances > self.tol_distance)

    @lazy_property.LazyProperty
    def false_negative_rate_with_tolerance(self):
        condition_positive = len(self.false_negative_distances)
        return float(self.false_negatives_with_tolerance) / float(condition_positive)

    @lazy_property.LazyProperty
    def true_positives_with_tolerance(self):
        all_pos = np.sum(np.invert(self.test_mask & self.truth_mask))
        return (
            all_pos
            - self.false_negatives_with_tolerance
            - self.false_positives_with_tolerance
        )

    @lazy_property.LazyProperty
    def precision_with_tolerance(self):
        return float(self.true_positives_with_tolerance) / float(
            self.true_positives_with_tolerance + self.false_positives_with_tolerance
        )

    @lazy_property.LazyProperty
    def recall_with_tolerance(self):
        return float(self.true_positives_with_tolerance) / float(
            self.true_positives_with_tolerance + self.false_negatives_with_tolerance
        )

    @lazy_property.LazyProperty
    def f1_score_with_tolerance(self):
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
        mean_false_positive_distance_clipped = np.mean(
            np.clip(self.false_positive_distances, None, self.clip_distance)
        )
        return mean_false_positive_distance_clipped

    @lazy_property.LazyProperty
    def mean_false_negative_distances_clipped(self):
        mean_false_negative_distance_clipped = np.mean(
            np.clip(self.false_negative_distances, None, self.clip_distance)
        )
        return mean_false_negative_distance_clipped

    @lazy_property.LazyProperty
    def mean_false_positive_distance(self):
        mean_false_positive_distance = np.mean(self.false_positive_distances)
        return mean_false_positive_distance

    @lazy_property.LazyProperty
    def false_negative_distances(self):
        truth_bin = np.invert(self.truth_mask)
        false_negative_distances = self.test_edt[truth_bin]
        return false_negative_distances

    @lazy_property.LazyProperty
    def mean_false_negative_distance(self):
        mean_false_negative_distance = np.mean(self.false_negative_distances)
        return mean_false_negative_distance

    @lazy_property.LazyProperty
    def mean_false_distance(self):
        mean_false_distance = 0.5 * (
            self.mean_false_positive_distance + self.mean_false_negative_distance
        )
        return mean_false_distance

    @lazy_property.LazyProperty
    def mean_false_distance_clipped(self):
        mean_false_distance_clipped = 0.5 * (
            self.mean_false_positive_distances_clipped
            + self.mean_false_negative_distances_clipped
        )
        return mean_false_distance_clipped

from typing import List
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray

from .evaluator import Evaluator
from .instance_evaluation_scores import InstanceEvaluationScores
from dacapo.utils.voi import voi as _voi

import numpy as np
import numpy_indexed as npi

import logging

logger = logging.getLogger(__name__)


def relabel(array, return_backwards_map=False, inplace=False):
    if array.size == 0:
        if return_backwards_map:
            return array, 0, []
        else:
            return array, 0

    # get all labels except 0
    old_labels = np.unique(array)
    old_labels = old_labels[old_labels != 0]

    if old_labels.size == 0:
        if return_backwards_map:
            return array, 0, [0]
        else:
            return array, 0

    n = len(old_labels)
    new_labels = np.arange(1, n + 1, dtype=array.dtype)

    replaced = npi.remap(
        array.flatten(), old_labels, new_labels, inplace=inplace
    ).reshape(array.shape)

    if return_backwards_map:
        backwards_map = np.insert(old_labels, 0, 0)
        return replaced, n, backwards_map

    return replaced, n


class InstanceEvaluator(Evaluator):
    criteria: List[str] = ["voi_merge", "voi_split", "voi"]

    def evaluate(self, output_array_identifier, evaluation_array):
        output_array = ZarrArray.open_from_array_identifier(output_array_identifier)
        evaluation_data = evaluation_array[evaluation_array.roi].astype(np.uint64)
        output_data = output_array[output_array.roi].astype(np.uint64)
        results = voi(evaluation_data, output_data)

        return InstanceEvaluationScores(
            voi_merge=results["voi_merge"],
            voi_split=results["voi_split"],
        )

    @property
    def score(self) -> InstanceEvaluationScores:
        return InstanceEvaluationScores()


def voi(truth, test):
    voi_split, voi_merge = _voi(test + 1, truth + 1, ignore_groundtruth=[])
    return {"voi_split": voi_split, "voi_merge": voi_merge}

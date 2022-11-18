import xarray as xr
from dacapo.experiments.datasplits.datasets.arrays import IntensitiesArray
import numpy as np


from .evaluator import Evaluator
from .instance_evaluation_scores import IntensitiesEvaluationScores

from funlib.evaluate import rand_voi

import random


class IntensitiesEvaluator(Evaluator):
    """IntensitiesEvaluator Class

    An evaluator takes a post-processor's output and compares it against
    ground-truth.
    """

    def evaluate(self, output_array_identifier, evaluation_array):
        output_array = IntensitiesArray.open_from_array_identifier(output_array_identifier)
        evaluation_data = evaluation_array[evaluation_array.roi].astype(np.uint64)
        output_data = output_array[output_array.roi].astype(np.uint64)
        results = rand_voi(evaluation_data, output_data)

        return IntensitiesEvaluationScores(
            voi_merge=results["voi_merge"], voi_split=results["voi_split"]
        )

    @property
    def score(self) -> IntensitiesEvaluationScores:
        return IntensitiesEvaluationScores()

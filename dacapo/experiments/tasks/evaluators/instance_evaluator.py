from dacapo.experiments.datasplits.datasets.arrays import ZarrArray

from .evaluator import Evaluator
from .instance_evaluation_scores import InstanceEvaluationScores

from funlib.evaluate import rand_voi, detection_scores

import numpy as np


class InstanceEvaluator(Evaluator):
    criteria = ["voi_merge", "voi_split", "voi"]

    def evaluate(self, output_array_identifier, evaluation_array):
        output_array = ZarrArray.open_from_array_identifier(output_array_identifier)
        evaluation_data = evaluation_array[evaluation_array.roi].astype(np.uint64)
        output_data = output_array[output_array.roi].astype(np.uint64)
        results = rand_voi(evaluation_data, output_data)
        results.update(
            detection_scores(
                evaluation_data,
                output_data,
                matching_score="iou",
                voxel_size=output_array.voxel_size,
            )
        )

        return InstanceEvaluationScores(
            voi_merge=results["voi_merge"],
            voi_split=results["voi_split"],
            avg_iou=results["avg_iou"],
        )

    @property
    def score(self) -> InstanceEvaluationScores:
        return InstanceEvaluationScores()

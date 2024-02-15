from dacapo.experiments.datasplits.datasets.arrays import ZarrArray

from .evaluator import Evaluator
from .instance_evaluation_scores import InstanceEvaluationScores

from funlib.evaluate import rand_voi

import numpy as np


class InstanceEvaluator(Evaluator):
    """
    InstanceEvaluator is an evaluator that computes scores for instance 
    segmentation tasks using Variation of Information (VOI) metrics.

    It calculates two key metrics: [VOI merge] and [VOI split], to evaluate the quality of instance 
    segmentation. These metrics are particularly useful for comparing the segmentation of objects 
    where each instance is uniquely labeled.

    Attributes:
        criteria (list): A list of criteria names used for evaluation. Defaults to 
                         ["voi_merge", "voi_split", "voi"].
    """
    criteria = ["voi_merge", "voi_split", "voi"]

    def evaluate(self, output_array_identifier, evaluation_array):
        """
        Evaluates the segmentation quality by computing VOI metrics.

        This method opens the output array from a given identifier, retrieves the relevant data
        from both output and evaluation arrays, and computes the VOI metrics.

        Args:
            output_array_identifier: An identifier for the Zarr array containing the output data.
            evaluation_array: An array containing the ground truth data for evaluation.

        Returns:
            InstanceEvaluationScores: An object containing the calculated VOI merge and split scores.
        """
        output_array = ZarrArray.open_from_array_identifier(output_array_identifier)
        evaluation_data = evaluation_array[evaluation_array.roi].astype(np.uint64)
        output_data = output_array[output_array.roi].astype(np.uint64)
        results = rand_voi(evaluation_data, output_data)

        return InstanceEvaluationScores(
            voi_merge=results["voi_merge"], voi_split=results["voi_split"]
        )

    @property
    def score(self) -> InstanceEvaluationScores:
        """
        A property that returns the evaluation scores.

        Note: This implementation currently returns an empty InstanceEvaluationScores object.
        This should be overridden to return the actual scores computed from the evaluate method.

        Returns:
            InstanceEvaluationScores: An object representing the evaluation scores.
        """
        return InstanceEvaluationScores()

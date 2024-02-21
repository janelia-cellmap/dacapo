from dacapo.experiments.datasplits.datasets.arrays import ZarrArray

from .evaluator import Evaluator
from .instance_evaluation_scores import InstanceEvaluationScores

from funlib.evaluate import rand_voi

import numpy as np


class InstanceEvaluator(Evaluator):
    """
    A subclass of Evaluator that specifically evaluates instance segmentation tasks. This class
    extends the base Evaluator class from dacapo library.

    Attributes:
        criteria (list[str]): A list of metric names that are used in this evaluation process.
    """

    criteria = ["voi_merge", "voi_split", "voi"]

    def evaluate(self, output_array_identifier, evaluation_array):
        """
        Evaluate the segmentation predictions with the ground truth data.

        Args:
            output_array_identifier: A unique id that refers to the array that contains
                                     predicted labels from the segmentation.
            evaluation_array: The ground truth labels to compare the predicted labels with.

        Returns:
            InstanceEvaluationScores: An object that includes the segmentation evaluation results.
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
        Property that returns the evaluation scores. However, currently, it only returns
        an empty InstanceEvaluationScores object.

        Returns:
            InstanceEvaluationScores: An object that supposedly contains evaluation scores.
        """
        return InstanceEvaluationScores()

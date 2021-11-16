from .evaluator import Evaluator
from .binary_segmentation_evaluation_scores import BinarySegmentationEvaluationScores


class BinarySegmentationEvaluator(Evaluator):
    def evaluate(self, output_array, evaluation_dataset):

        return BinarySegmentationEvaluationScores(
            frizz_level=9.0,
        )

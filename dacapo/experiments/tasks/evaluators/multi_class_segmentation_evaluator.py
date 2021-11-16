from .evaluator import Evaluator
from .multi_class_segmentation_evaluation_scores import MultiClassSegmentationEvaluationScores


class MultiClassSegmentationEvaluator(Evaluator):
    def evaluate(self, output_array, evaluation_dataset):

        return MultiClassSegmentationEvaluationScores(
            frizz_level=9.0,
        )

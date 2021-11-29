from .evaluator import Evaluator
from .multi_class_segmentation_evaluation_scores import MultiClassSegmentationEvaluationScores


class MultiClassSegmentationEvaluator(Evaluator):
    def evaluate(self, output_array, evaluation_dataset):

        return MultiClassSegmentationEvaluationScores(
            frizz_level=9.0,
        )

    def is_best(self, iteration, parameter, score, criterion):
        """
        Check if the provided score is the best according to some criterion
        """
        return []

    def set_best(self, iteration_scores):
        pass
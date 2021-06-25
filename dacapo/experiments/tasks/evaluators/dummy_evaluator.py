from .evaluator import Evaluator
from .dummy_evaluation_scores import DummyEvaluationScores


class DummyEvaluator(Evaluator):

    def evaluate(
            self,
            output_array,
            evaluation_dataset):

        return DummyEvaluationScores(
            frizz_level=9.0,
            blipp_score=0.03)

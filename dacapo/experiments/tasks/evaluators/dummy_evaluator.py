from .evaluator import Evaluator
from .dummy_evaluation_scores import DummyEvaluationScores


class DummyEvaluator(Evaluator):
    criteria = ["frizz_level", "blipp_score"]

    def evaluate(self, output_array, evaluation_dataset):

        return DummyEvaluationScores(frizz_level=9.0, blipp_score=0.03)

    def is_best(self, iteration, parameters, score, criterion):
        """
        Check if the provided score is the best according to some criterion
        """
        return self.criteria

    def set_best(self, iteration_scores):
        pass

    @property
    def score(self) -> DummyEvaluationScores:
        return DummyEvaluationScores()
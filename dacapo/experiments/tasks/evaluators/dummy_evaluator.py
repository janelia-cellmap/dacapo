from .evaluator import Evaluator
from .dummy_evaluation_scores import DummyEvaluationScores

import random


class DummyEvaluator(Evaluator):
    criteria = ["frizz_level", "blipp_score"]

    def evaluate(self, output_array_identifier, evaluation_dataset):
        """
        Evaluate the given output array and dataset and returns the scores based on predefined criteria.

        Args:
            output_array_identifier : The output array to be evaluated.
            evaluation_dataset : The dataset to be used for evaluation.

        Returns:
            DummyEvaluationScore: An object of DummyEvaluationScores class, with the evaluation scores.
        """
        return DummyEvaluationScores(
            frizz_level=random.random(), blipp_score=random.random()
        )

    @property
    def score(self) -> DummyEvaluationScores:
        return DummyEvaluationScores()

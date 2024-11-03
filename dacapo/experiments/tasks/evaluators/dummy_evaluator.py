from .evaluator import Evaluator
from .dummy_evaluation_scores import DummyEvaluationScores

import random


class DummyEvaluator(Evaluator):
    

    criteria = ["frizz_level", "blipp_score"]

    def evaluate(self, output_array_identifier, evaluation_dataset):
        
        return DummyEvaluationScores(
            frizz_level=random.random(), blipp_score=random.random()
        )

    @property
    def score(self) -> DummyEvaluationScores:
        
        return DummyEvaluationScores()

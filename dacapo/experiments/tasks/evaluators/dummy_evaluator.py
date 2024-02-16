```python
from .evaluator import Evaluator
from .dummy_evaluation_scores import DummyEvaluationScores

import random

class DummyEvaluator(Evaluator):
    """
    A Dummy Evaluator class which extends the Evaluator class for evaluation operations.
    
    Attributes:
        criteria (list): List of evaluation criteria.
    """
    criteria = ["frizz_level", "blipp_score"]

    def evaluate(self, output_array, evaluation_dataset):
        """
        Evaluate the given output array and dataset and returns the scores based on predefined criteria.
        
        Args:
            output_array : The output array to be evaluated.
            evaluation_dataset : The dataset to be used for evaluation.
        
        Returns:
            DummyEvaluationScore: An object of DummyEvaluationScores class, with the evaluation scores.
        """
        return DummyEvaluationScores(
            frizz_level=random.random(), blipp_score=random.random()
        )

    @property
    def score(self) -> DummyEvaluationScores:
        """
        A property which is the instance of DummyEvaluationScores containing the evaluation scores.

        Returns:
            DummyEvaluationScores: An object of DummyEvaluationScores class.
        """
        return DummyEvaluationScores()
```
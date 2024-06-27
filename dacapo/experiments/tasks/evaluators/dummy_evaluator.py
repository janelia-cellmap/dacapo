from .evaluator import Evaluator
from .dummy_evaluation_scores import DummyEvaluationScores

import random


class DummyEvaluator(Evaluator):
    """
    A class representing a dummy evaluator. This evaluator is used for testing purposes.

    Attributes:
        criteria : List[str]
            the evaluation criteria
    Methods:
        evaluate(output_array_identifier, evaluation_dataset)
            Evaluate the output array against the evaluation dataset.
        score
            Return the evaluation scores.
    Note:
        The DummyEvaluator class is used to evaluate the performance of a dummy task.
    """

    criteria = ["frizz_level", "blipp_score"]

    def evaluate(self, output_array_identifier, evaluation_dataset):
        """
        Evaluate the given output array and dataset and returns the scores based on predefined criteria.

        Args:
            output_array_identifier : The output array to be evaluated.
            evaluation_dataset : The dataset to be used for evaluation.
        Returns:
            DummyEvaluationScore: An object of DummyEvaluationScores class, with the evaluation scores.
        Raises:
            ValueError: if the output array identifier is not valid
        Examples:
            >>> dummy_evaluator = DummyEvaluator()
            >>> output_array_identifier = "output_array"
            >>> evaluation_dataset = "evaluation_dataset"
            >>> dummy_evaluator.evaluate(output_array_identifier, evaluation_dataset)
            DummyEvaluationScores(frizz_level=0.0, blipp_score=0.0)
        Note:
            This function is used to evaluate the output array against the evaluation dataset.
        """
        return DummyEvaluationScores(
            frizz_level=random.random(), blipp_score=random.random()
        )

    @property
    def score(self) -> DummyEvaluationScores:
        """
        Return the evaluation scores.

        Returns:
            DummyEvaluationScores: An object of DummyEvaluationScores class, with the evaluation scores.
        Examples:
            >>> dummy_evaluator = DummyEvaluator()
            >>> dummy_evaluator.score
            DummyEvaluationScores(frizz_level=0.0, blipp_score=0.0)
        Note:
            This function is used to return the evaluation scores.
        """
        return DummyEvaluationScores()

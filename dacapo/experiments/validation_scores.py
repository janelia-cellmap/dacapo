from .validation_iteration_scores import ValidationIterationScores
from typing import List
import attr


@attr.s
class ValidationScores:

    iteration_scores: List[ValidationIterationScores] = attr.ib(
        default=attr.Factory(list),
        metadata={"help_text": "An ordered list of validation scores by iteration."},
    )

    def add_iteration_scores(self, iteration_scores):

        self.iteration_scores.append(iteration_scores)

    def delete_after(self, iteration):

        self.iteration_scores = [
            scores for scores in self.iteration_scores if scores.iteration < iteration
        ]

    def validated_until(self):
        """The number of iterations validated for (the maximum iteration plus
        one)."""

        if not self.iteration_scores:
            return 0
        return max([score.iteration for score in self.iteration_scores]) + 1


from .tasks.evaluators import EvaluationScores
from .tasks.post_processors import PostProcessorParameters
from typing import List, Tuple
import attr


@attr.s
class ValidationIterationScores:

    iteration: int = attr.ib(
        metadata={"help_text": "The iteration associated with these validation scores."}
    )
    parameter_scores: List[Tuple[PostProcessorParameters, EvaluationScores]] = attr.ib(
        metadata={
            "help_text": "A list of evaluation scores and their associated post-processing parameters."
        }
    )

    @property
    def scores(self):
        return [score for params, score in self.parameter_scores]

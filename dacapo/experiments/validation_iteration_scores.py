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
    dataset: str = attr.ib(
        default=None, metadata={"help_text": "The dataset that these stats belong to."}
    )

    @property
    def scores(self):
        return [score for params, score in self.parameter_scores]

    @property
    def criteria(self):
        if self.parameter_scores:
            return self.parameter_scores[0][1].criteria

        raise RuntimeError("No scores to evaluate yet")

    @property
    def parameter_names(self):
        if self.parameter_scores:
            postprocessor_class_instance = self.parameter_scores[0][0]
            return postprocessor_class_instance.parameter_names

        raise RuntimeError("No scores to evaluate yet")

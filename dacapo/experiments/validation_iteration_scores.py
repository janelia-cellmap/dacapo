from .tasks.evaluators import EvaluationScores
from .tasks.post_processors import PostProcessorParameters
from typing import List, Tuple
import attr


@attr.s
class ValidationIterationScores:

    iteration: int = attr.ib()
    parameter_scores: List[Tuple[PostProcessorParameters, EvaluationScores]] = attr.ib()

    def is_better(self, reference_scores, score_name, lower_is_better):
        """Return True if the given reference score is better than the already
        stored scores, in terms of the given score name."""

        reference_score = getattr(reference_scores, score_name)

        for _, compare_scores in self.parameter_scores:

            compare_score = getattr(compare_scores, score_name)

            if lower_is_better:

                if compare_score <= reference_score:
                    return False

            else:

                if compare_score >= reference_score:
                    return False

        return True

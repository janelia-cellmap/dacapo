from .tasks.evaluators import EvaluationScores
from .tasks.post_processors import PostProcessorParameters
from typing import List, Tuple
import attr


@attr.s
class ValidationIterationScores:

    iteration: int = attr.ib(
        metadata={"help_text": "The iteration associated with these validation scores."}
    )
    scores: List[List[float]] = attr.ib(
        metadata={
            "help_text": "A list of scores per post processor parameters and evaluation criterion."
        }
    )

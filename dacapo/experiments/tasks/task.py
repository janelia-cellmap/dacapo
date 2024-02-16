from .predictors import Predictor
from .losses import Loss
from .evaluators import Evaluator, EvaluationScores
from .post_processors import PostProcessor, PostProcessorParameters

from abc import ABC
from typing import Iterable


class Task(ABC):
    predictor: Predictor
    loss: Loss
    evaluator: Evaluator
    post_processor: PostProcessor

    @property
    def parameters(self) -> Iterable[PostProcessorParameters]:
        return list(self.post_processor.enumerate_parameters())

    @property
    def evaluation_scores(self) -> EvaluationScores:
        return self.evaluator.score

    def create_model(self, architecture):
        return self.predictor.create_model(architecture=architecture)

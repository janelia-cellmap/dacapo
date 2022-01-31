from .predictors import Predictor
from .losses import Loss
from .evaluators import Evaluator, EvaluationScores
from .post_processors import PostProcessor, PostProcessorParameters

from abc import ABC, abstractmethod
from typing import Iterable


class Task(ABC):
    @property
    @abstractmethod
    def predictor(self) -> Predictor:
        """The predictors to use on for this task"""
        pass

    @property
    @abstractmethod
    def loss(self) -> Loss:
        pass

    @property
    @abstractmethod
    def evaluator(self) -> Evaluator:
        pass

    @property
    @abstractmethod
    def post_processor(self) -> PostProcessor:
        pass

    @property
    def parameters(self) -> Iterable[PostProcessorParameters]:
        return list(self.post_processor.enumerate_parameters())

    @property
    def evaluation_scores(self) -> EvaluationScores:
        return self.evaluator.score

    def create_model(self, architecture):
        return self.predictor.create_model(architecture=architecture)

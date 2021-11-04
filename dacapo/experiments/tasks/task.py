from .predictors import Predictor
from .losses import Loss
from .evaluators import Evaluator
from .post_processors import PostProcessor

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

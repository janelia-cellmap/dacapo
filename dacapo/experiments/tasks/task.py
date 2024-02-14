from .predictors import Predictor
from .losses import Loss
from .evaluators import Evaluator, EvaluationScores
from .post_processors import PostProcessor, PostProcessorParameters

from abc import ABC
from typing import Iterable

class Task(ABC):
    """
    Abstract base class for tasks in a machine learning or data processing pipeline.

    This class provides a structure for tasks that involve prediction, loss calculation,
    evaluation, and post-processing. It is designed to be extended by specific task
    implementations that define the behavior of these components.

    Attributes:
        predictor (Predictor): An instance of a Predictor, responsible for making predictions.
        loss (Loss): An instance of a Loss, used for calculating the loss of the model.
        evaluator (Evaluator): An instance of an Evaluator, used for evaluating the model's performance.
        post_processor (PostProcessor): An instance of a PostProcessor, used for processing the output of the model.
    """

    predictor: Predictor
    loss: Loss
    evaluator: Evaluator
    post_processor: PostProcessor

    @property
    def parameters(self) -> Iterable[PostProcessorParameters]:
        """
        A property that returns an iterable of post-processor parameters.

        This method enumerates through the parameters of the post_processor attribute
        and returns them in a list.

        Returns:
            Iterable[PostProcessorParameters]: An iterable collection of post-processor parameters.
        """
        return list(self.post_processor.enumerate_parameters())

    @property
    def evaluation_scores(self) -> EvaluationScores:
        """
        A property that returns the evaluation scores.

        This method accesses the score attribute of the evaluator to provide an 
        assessment of the model's performance.

        Returns:
            EvaluationScores: An object representing the evaluation scores of the model.
        """
        return self.evaluator.score

    def create_model(self, architecture):
        """
        Creates a model based on the specified architecture.

        This method utilizes the predictor's method to create a model with the given architecture. 
        It abstracts the model creation process, allowing different implementations based on the 
        predictor's type.

        Args:
            architecture: The architecture specification for the model to be created.

        Returns:
            A model instance created based on the specified architecture.
        """
        return self.predictor.create_model(architecture=architecture)


from .evaluators import DummyEvaluator
from .losses import DummyLoss
from .post_processors import ArgmaxPostProcessor
from .predictors import OneHotPredictor
from .task import Task

import warnings


class OneHotTask(Task):
    """
    A task that uses a one-hot predictor. The model is loaded from a file
    and the weights are loaded from a file. The loss is a dummy loss and the
    post processor is an argmax post processor. The evaluator is a dummy evaluator.

    Attributes:
        weights (Path): The path to the weights file.
    Methods:
        create_model(self, architecture) -> Model: This method creates a model using the given architecture.
    Notes:
        This is a base class for all tasks that use one-hot predictors.
    """

    def __init__(self, task_config):
        """
        Initialize the OneHotTask object.

        Args:
            task_config (TaskConfig): The configuration of the task.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> task = OneHotTask(task_config)
        """

        if task_config.kernel_size is None:
            warnings.warn(
                "The default kernel size of 3 will be changing to 1. "
                "Please specify the kernel size explicitly.",
                DeprecationWarning,
            )
            task_config.kernel_size = 3
        self.predictor = OneHotPredictor(
            classes=task_config.classes, kernel_size=task_config.kernel_size
        )
        self.loss = DummyLoss()
        self.post_processor = ArgmaxPostProcessor()
        self.evaluator = DummyEvaluator()

        self._classes = task_config.classes

    @property
    def channels(self) -> list[str]:
        return self._classes

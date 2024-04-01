from .evaluators import DummyEvaluator
from .losses import DummyLoss
from .post_processors import ArgmaxPostProcessor
from .predictors import OneHotPredictor
from .task import Task


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
        self.predictor = OneHotPredictor(classes=task_config.classes)
        self.loss = DummyLoss()
        self.post_processor = ArgmaxPostProcessor()
        self.evaluator = DummyEvaluator()

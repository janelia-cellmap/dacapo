from .evaluators import DummyEvaluator
from .losses import DummyLoss
from .post_processors import ArgmaxPostProcessor
from .predictors import OneHotPredictor
from .task import Task


class OneHotTask(Task):
    """
    OneHotTask is a specialized implementation of a Task that performs one-hot encoding 
    for a given set of classes. It integrates various components like a predictor, loss function, 
    post-processor, and evaluator, which are configured based on the provided task configuration.

    Attributes:
        predictor (OneHotPredictor): An instance of OneHotPredictor initialized with the specified classes.
        loss (DummyLoss): An instance of DummyLoss, a placeholder for loss computation.
        post_processor (ArgmaxPostProcessor): An instance of ArgmaxPostProcessor for post-processing predictions.
        evaluator (DummyEvaluator): An instance of DummyEvaluator for evaluating the task performance.
    """

    def __init__(self, task_config):
        """
        Initializes a new instance of the OneHotTask class.

        Args:
            task_config: A configuration object specific to the task. It must contain a 'classes'
                         attribute which is used to initialize the OneHotPredictor.

        The constructor initializes four main components of the task:
        - predictor: A OneHotPredictor that is initialized with the classes from the task configuration.
        - loss: A DummyLoss instance, representing a placeholder for the actual loss computation.
        - post_processor: An ArgmaxPostProcessor, which post-processes the predictions.
        - evaluator: A DummyEvaluator, used for evaluating the task's performance.
        """
        self.predictor = OneHotPredictor(classes=task_config.classes)
        self.loss = DummyLoss()
        self.post_processor = ArgmaxPostProcessor()
        self.evaluator = DummyEvaluator()


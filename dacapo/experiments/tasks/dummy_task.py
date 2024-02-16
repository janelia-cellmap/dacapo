from .evaluators import DummyEvaluator
from .losses import DummyLoss
from .post_processors import DummyPostProcessor
from .predictors import DummyPredictor
from .task import Task


class DummyTask(Task):
    """
    A dummy task class that initializes all components (predictor, loss, 
    post-processing, and evaluator) for the dummy task. Primarily used for testing purposes. 
    Inherits from the Task class.

    Attributes
    ----------
    predictor : Object
        Instance of DummyPredictor class.
    loss : Object
        Instance of DummyLoss class.
    post_processor : Object
        Instance of DummyPostProcessor class.
    evaluator : Object
        Instance of DummyEvaluator class.
    """

    def __init__(self, task_config):
        """
        Initializes dummy task with predictor, loss function, post processor and evaluator.

        Parameters
        ----------
        task_config : Object
            Configurations for the task, contains `embedding_dims` and `detection_threshold`
        """

        self.predictor = DummyPredictor(task_config.embedding_dims)
        self.loss = DummyLoss()
        self.post_processor = DummyPostProcessor(task_config.detection_threshold)
        self.evaluator = DummyEvaluator()
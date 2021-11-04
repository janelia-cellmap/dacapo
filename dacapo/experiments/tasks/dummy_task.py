from .evaluators import DummyEvaluator
from .losses import DummyLoss
from .post_processors import DummyPostProcessor
from .predictors import DummyPredictor
from .task import Task


class DummyTask(Task):
    """This is just a dummy task for testing."""

    predictor = None
    loss = None
    post_processor = None
    evaluator = None

    def __init__(self, task_config):
        """Create a `DummyTask` from a `DummyTaskConfig`."""

        self.predictor = DummyPredictor(task_config.embedding_dims)
        self.loss = DummyLoss()
        self.post_processor = DummyPostProcessor(task_config.detection_threshold)
        self.evaluator = DummyEvaluator()


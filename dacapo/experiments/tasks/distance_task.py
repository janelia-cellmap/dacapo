from .evaluators import BinarySegmentationEvaluator
from .losses import DummyLoss
from .post_processors import ThresholdPostProcessor
from .predictors import DistancePredictor
from .task import Task


class DistanceTask(Task):
    """This is just a dummy task for testing."""

    predictor = None
    loss = None
    post_processor = None
    evaluator = None

    def __init__(self, task_config):
        """Create a `DummyTask` from a `DummyTaskConfig`."""

        self.predictor = DistancePredictor(
            channels=task_config.channels, scale_factor=task_config.scale_factor
        )
        self.loss = DummyLoss()
        self.post_processor = ThresholdPostProcessor()
        self.evaluator = BinarySegmentationEvaluator()

from .evaluators import BinarySegmentationEvaluator
from .losses import MSELoss
from .post_processors import ThresholdPostProcessor
from .predictors import InnerDistancePredictor
from .task import Task


# Goal is have a distance task but with distance inside the forground only
class InnerDistanceTask(Task):
    """This is just a dummy task for testing."""

    def __init__(self, task_config):
        """Create a `DummyTask` from a `DummyTaskConfig`."""

        self.predictor = InnerDistancePredictor(
            channels=task_config.channels,
            scale_factor=task_config.scale_factor,
        )
        self.loss = MSELoss()
        self.post_processor = ThresholdPostProcessor()
        self.evaluator = BinarySegmentationEvaluator(
            clip_distance=task_config.clip_distance,
            tol_distance=task_config.tol_distance,
            channels=task_config.channels,
        )

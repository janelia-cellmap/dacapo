from .losses import DummyLoss
from .post_processors import DummyPostProcessor
from .predictors import DummyPredictor
from .task import Task


class DummyTask(Task):
    """This is just a dummy task for testing."""

    def __init__(self, task_config):
        """Create a `DummyTask` from a `DummyTaskConfig`."""

        predictor = DummyPredictor(task_config.embedding_dims)
        loss = DummyLoss()
        post_processor = DummyPostProcessor(task_config.detection_threshold)

        super().__init__(predictor, loss, post_processor)

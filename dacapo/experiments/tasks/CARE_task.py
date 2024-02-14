from .evaluators import IntensitiesEvaluator
from .losses import MSELoss
from .post_processors import CAREPostProcessor
from .predictors import CAREPredictor
from .task import Task


class CARETask(Task):
    """CAREPredictor."""

    def __init__(self, task_config) -> None:
        """Create a `CARETask`."""
        self.predictor = CAREPredictor(
            num_channels=task_config.num_channels, dims=task_config.dims
        )
        self.loss = MSELoss()
        self.post_processor = CAREPostProcessor()
        self.evaluator = IntensitiesEvaluator()

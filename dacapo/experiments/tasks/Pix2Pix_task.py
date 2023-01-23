from .evaluators import IntensitiesEvaluator
from .losses import MSELoss
from .post_processors import CAREPostProcessor
from .predictors import CAREPredictor
from .task import Task

class Pix2PixTask(Task):
    """Pix2Pix Predictor."""

    def __init__(self, task_config) -> None:
        """Create a `Pix2PixTask`."""
        self.predictor = Pix2Pix_predictor(num_channels=task_config.num_channels, dims=task_config.dims)
        self.loss = MSELoss()  # TODO: change losses
        self.post_processor = CAREPostProcessor()  # TODO: change post processor
        self.evaluator = IntensitiesEvaluator()
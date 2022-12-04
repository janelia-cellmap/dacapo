from .evaluators import IntensitiesEvaluator
from .losses import MSELoss
from .post_processors import CAREPostProcessor
from .predictors import CAREPredictor
from .task import Task


class CycleGANTask(Task):
    """CycleGAN Task."""

    def __init__(self, task_config) -> None:
        """Create a `CycleGAN Task`."""

        self.predictor = CycleGANPredictor(num_channels=task_config.num_channels)
        self.loss = LinkCycleLoss()
        self.post_processor = CycleGANPostProcessor()
        self.evaluator = IntensitiesEvaluator()

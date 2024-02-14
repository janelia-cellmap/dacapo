from .evaluators import IntensitiesEvaluator
from .losses import GANLoss
from .post_processors import CycleGANPostProcessor
from .predictors import CycleGANPredictor
from .task import Task


class CycleGANTask(Task):
    """CycleGAN Task."""

    def __init__(self, task_config) -> None:
        """Create a `CycleGAN Task`."""

        self.predictor = CycleGANPredictor(num_channels=task_config.num_channels)
        self.loss = GANLoss()
        self.post_processor = CycleGANPostProcessor()
        self.evaluator = IntensitiesEvaluator()

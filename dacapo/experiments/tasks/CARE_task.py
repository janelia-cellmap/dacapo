from .evaluators import IntensitiesEvaluator
from .evaluators import IntensitiesEvaluationScores
from .losses import MSELoss
from .post_processors import CAREPostProcessor
from .post_processors import CAREPostProcessorParameters
from .predictors import CAREPredictor
from .task import Task


class CARETask(Task):
    """This is a task for generating voxel affinities."""

    def __init__(self, task_config) -> None:
        """Create a `DummyTask` from a `DummyTaskConfig`."""

        self.predictor = CAREPredictor(
            neighborhood=task_config.neighborhood, lsds=task_config.lsds
        )
        self.loss = MSELoss(len(task_config.neighborhood))
        self.post_processor = CAREPostProcessor(offsets=task_config.neighborhood)
        self.evaluator = IntensitiesEvaluator()

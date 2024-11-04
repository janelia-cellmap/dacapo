from .evaluators import BinarySegmentationEvaluator
from .losses import MSELoss
from .post_processors import ThresholdPostProcessor
from .predictors import InnerDistancePredictor
from .task import Task


class InnerDistanceTask(Task):
    def __init__(self, task_config):
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

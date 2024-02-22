from .evaluators import BinarySegmentationEvaluator
from .losses import HotDistanceLoss
from .post_processors import ThresholdPostProcessor
from .predictors import HotDistancePredictor
from .task import Task


class HotDistanceTask(Task):
    """This is just a Hot Distance Task that combine Binary and distance prediction."""

    def __init__(self, task_config):
        """Create a `HotDistanceTask` from a `HotDistanceTaskConfig`."""

        self.predictor = HotDistancePredictor(
            channels=task_config.channels,
            scale_factor=task_config.scale_factor,
            mask_distances=task_config.mask_distances,
        )
        self.loss = HotDistanceLoss()
        self.post_processor = ThresholdPostProcessor()
        self.evaluator = BinarySegmentationEvaluator(
            clip_distance=task_config.clip_distance,
            tol_distance=task_config.tol_distance,
            channels=task_config.channels,
        )

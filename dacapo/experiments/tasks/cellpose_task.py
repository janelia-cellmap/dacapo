from .evaluators import InstanceEvaluator
from .losses import CellposeLoss
from .post_processors import CellposePostProcessor
from .predictors import CellposePredictor
from .task import Task


class CellposeTask(Task):
    def __init__(self, task_config):

        self.predictor = CellposePredictor(
            channels=task_config.channels,
            scale_factor=task_config.scale_factor,
            mask_distances=task_config.mask_distances,
            clipmin=task_config.clipmin,
            clipmax=task_config.clipmax,
        )
        self.loss = CellposeLoss()
        self.post_processor = ThresholdPostProcessor()
        self.evaluator = InstanceEvaluator()

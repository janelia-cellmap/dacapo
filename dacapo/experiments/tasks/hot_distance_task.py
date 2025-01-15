from .evaluators import BinarySegmentationEvaluator
from .losses import HotDistanceLoss
from .post_processors import ThresholdPostProcessor
from .predictors import HotDistancePredictor
from .task import Task

import warnings


class HotDistanceTask(Task):
    """
    A class to represent a hot distance task that use binary prediction and distance prediction.

    Inherits from Task class.

    Attributes:
        predictor: HotDistancePredictor object.
        loss: HotDistanceLoss object.
        post_processor: ThresholdPostProcessor object.
        evaluator: BinarySegmentationEvaluator object.
    Methods:
        __init__(self, task_config): Constructs all the necessary attributes for the HotDistanceTask object.
    Notes:
        This is a subclass of Task.
    """

    def __init__(self, task_config):
        """
        Constructs all the necessary attributes for the HotDistanceTask object.

        Args:
            task_config : The task configuration parameters.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> task = HotDistanceTask(task_config)

        """

        if task_config.kernel_size is None:
            warnings.warn(
                "The default kernel size of 3 will be changing to 1. "
                "Please specify the kernel size explicitly.",
                DeprecationWarning,
            )
            task_config.kernel_size = 3
        self.predictor = HotDistancePredictor(
            channels=task_config.channels,
            scale_factor=task_config.scale_factor,
            mask_distances=task_config.mask_distances,
            kernel_size=task_config.kernel_size,
        )
        self.loss = HotDistanceLoss()
        self.post_processor = ThresholdPostProcessor()
        self.evaluator = BinarySegmentationEvaluator(
            clip_distance=task_config.clip_distance,
            tol_distance=task_config.tol_distance,
            channels=task_config.channels,
        )

        self._channels = task_config.channels

    @property
    def channels(self) -> list[str]:
        return self._channels
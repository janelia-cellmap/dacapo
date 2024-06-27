from .evaluators import BinarySegmentationEvaluator
from .losses import MSELoss
from .post_processors import ThresholdPostProcessor
from .predictors import DistancePredictor
from .task import Task


class DistanceTask(Task):
    """DistanceTask is a subclass of Task for handling tasks associated
    with Distance.

    DistanceTask uses `DistancePredictor` for prediction, `MSELoss` for
    computing loss, `ThresholdPostProcessor` for post-processing the
    prediction, and `BinarySegmentationEvaluator` for evaluating the
    prediction.

    Attributes:
        predictor: DistancePredictor object
        loss: MSELoss object
        post_processor: ThresholdPostProcessor object
        evaluator: BinarySegmentationEvaluator object
    Methods:
        __init__(self, task_config): Initializes attributes of DistanceTask
    Notes:
        This is a subclass of Task.
    """

    def __init__(self, task_config):
        """Initializes attributes of DistanceTask.

        It initializes predictor, loss, post processor, and evaluator
        based on the controls provided in task_config.

        Args:
            task_config: Object of task configuration
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> task = DistanceTask(task_config)
        """

        self.predictor = DistancePredictor(
            channels=task_config.channels,
            scale_factor=task_config.scale_factor,
            mask_distances=task_config.mask_distances,
            clipmin=task_config.clipmin,
            clipmax=task_config.clipmax,
        )
        self.loss = MSELoss()
        self.post_processor = ThresholdPostProcessor()
        self.evaluator = BinarySegmentationEvaluator(
            clip_distance=task_config.clip_distance,
            tol_distance=task_config.tol_distance,
            channels=task_config.channels,
        )

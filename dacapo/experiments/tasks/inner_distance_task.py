from .evaluators import BinarySegmentationEvaluator
from .losses import MSELoss
from .post_processors import ThresholdPostProcessor
from .predictors import InnerDistancePredictor
from .task import Task


class InnerDistanceTask(Task):
    """This class extends the Task class for creating tasks related to computing inner distances.
    It provides methods for prediction, loss calculation and post-processing. It includes Binary Segmentation Evaluator for evaluation.

    Attributes:
        task_config: The configuration for the task.
        predictor: Used for predicting the inner distances.
        loss: Used for calculating the mean square error loss.
        post_processor: Used for applying threshold post-processing.
        evaluator: Used for evaluating the results using binary segmentation.
    Methods:
        __init__(self, task_config): Initializes an instance of InnerDistanceTask.
    Notes:
        This is a subclass of Task.
    """

    def __init__(self, task_config):
        """
        Initializes an instance of InnerDistanceTask.

        Args:
            task_config: The configuration for the task including channel and scale factor for prediction,
                         and clip distance, tolerance distance, and channels for evaluation.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> task = InnerDistanceTask(task_config)

        """

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

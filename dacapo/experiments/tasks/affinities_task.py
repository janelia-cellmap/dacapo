from .evaluators import InstanceEvaluator
from .losses import AffinitiesLoss
from .post_processors import WatershedPostProcessor
from .predictors import AffinitiesPredictor
from .task import Task


class AffinitiesTask(Task):
    """
    This is a task for generating voxel affinities. It uses an `AffinitiesPredictor` for prediction,
    an `AffinitiesLoss` for loss calculation, a `WatershedPostProcessor` for post-processing, and an
    `InstanceEvaluator` for evaluation.

    Attributes:
        predictor: AffinitiesPredictor object
        loss: AffinitiesLoss object
        post_processor: WatershedPostProcessor object
        evaluator: InstanceEvaluator object
    Methods:
        __init__(self, task_config): Initializes all components for the affinities task.
    Notes:
        This is a subclass of Task.

    """

    def __init__(self, task_config):
        """
        Create a `DummyTask` from a `DummyTaskConfig`.

        Args:
            task_config: The configuration for the task.
        Returns:
            A `DummyTask` object.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> task = AffinitiesTask(task_config)
        """

        self.predictor = AffinitiesPredictor(
            neighborhood=task_config.neighborhood,
            lsds=task_config.lsds,
            affs_weight_clipmin=task_config.affs_weight_clipmin,
            affs_weight_clipmax=task_config.affs_weight_clipmax,
            lsd_weight_clipmin=task_config.lsd_weight_clipmin,
            lsd_weight_clipmax=task_config.lsd_weight_clipmax,
            background_as_object=task_config.background_as_object,
        )
        self.loss = AffinitiesLoss(
            len(task_config.neighborhood), task_config.lsds_to_affs_weight_ratio
        )
        self.post_processor = WatershedPostProcessor(offsets=task_config.neighborhood)
        self.evaluator = InstanceEvaluator()

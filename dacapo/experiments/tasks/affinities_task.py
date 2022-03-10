from .evaluators import InstanceEvaluator
from .losses import MSELoss
from .post_processors import WatershedPostProcessor
from .predictors import AffinitiesPredictor
from .task import Task


class AffinitiesTask(Task):
    """This is a task for generating voxel affinities."""

    predictor = None
    loss = None
    post_processor = None
    evaluator = None

    def __init__(self, task_config):
        """Create a `DummyTask` from a `DummyTaskConfig`."""

        self.predictor = AffinitiesPredictor(
            neighborhood=task_config.neighborhood, lsds=task_config.lsds
        )
        self.loss = MSELoss()
        self.post_processor = WatershedPostProcessor(offsets=task_config.neighborhood)
        self.evaluator = InstanceEvaluator()

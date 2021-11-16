from .evaluators import MultiClassSegmentationEvaluator
from .losses import DummyLoss
from .post_processors import ArgmaxPostProcessor
from .predictors import OneHotPredictor
from .task import Task


class OneHotTask(Task):

    predictor = None
    loss = None
    post_processor = None
    evaluator = None

    def __init__(self, task_config):
        self.predictor = OneHotPredictor(classes=task_config.classes)
        self.loss = DummyLoss()
        self.post_processor = ArgmaxPostProcessor()
        self.evaluator = MultiClassSegmentationEvaluator()

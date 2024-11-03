from .evaluators import DummyEvaluator
from .losses import DummyLoss
from .post_processors import ArgmaxPostProcessor
from .predictors import OneHotPredictor
from .task import Task


class OneHotTask(Task):
    def __init__(self, task_config):
        self.predictor = OneHotPredictor(classes=task_config.classes)
        self.loss = DummyLoss()
        self.post_processor = ArgmaxPostProcessor()
        self.evaluator = DummyEvaluator()

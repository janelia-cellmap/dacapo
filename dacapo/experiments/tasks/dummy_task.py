from .evaluators import DummyEvaluator
from .losses import DummyLoss
from .post_processors import DummyPostProcessor
from .predictors import DummyPredictor
from .task import Task


class DummyTask(Task):
    

    def __init__(self, task_config):
        

        self.predictor = DummyPredictor(task_config.embedding_dims)
        self.loss = DummyLoss()
        self.post_processor = DummyPostProcessor(task_config.detection_threshold)
        self.evaluator = DummyEvaluator()
